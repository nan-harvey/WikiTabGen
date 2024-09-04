# %%
import openai
import logging
import tiktoken

import pandas as pd
import json
from dataclasses import dataclass
from dateutil import parser as date_parser
from unidecode import unidecode
from config import setup_rich_logging

import os
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.progress import Progress
from utils.openai_util import calculate_max_output_tokens
from utils.shared_ai_util import CompletionModel
from utils.table_utils import format_answer_text_with_table, find_markdown_tables
from pathlib import Path


TEMPLATE_FULL_TABLE_MD = """
You are a retriever of facts.
We want to create a table with the detailed information about %s.
Each element of the response will contain %d fields: %s.
""" 

TEMPLATE_FULL_TABLE_JSON = """
You are a retriever of facts.
We want to create a table with the detailed information about %s.
Each element of the response will contain %d fields: %s.

Format all tables as follows:

```json_column_headers
["header1", "header2", "header3"]
```

```json_table_data
[
    ["cell1.1", "cell1.2", "cell1.3"],
    ["cell2.1", "cell2.2", "cell2.3"],
    ["cell3.1", "cell3.2", "cell3.3"],
]
```
""" 

BASE_PATH = Path(__file__).parent
SAVE_PATH = BASE_PATH / "results"
DATA_PATH = BASE_PATH / "benchmark"

class TableGenerator():

    def _norm_field(self, s):
        s = s.lower().replace(" ","_").replace("-","_").replace(".", "").replace(",","_")\
                .replace("(", "").replace(")", "").replace(":", "").replace('"','').replace("'","")\
                .replace("/", "")
        return re.sub('_+', '_', s)
    
    def generate_prompt(self, template, table_description, fields):
        num_fields = len(fields)
        fields_json = []
        fields = [self._norm_field(f) for f in fields]
        for field in fields:
            fields_json.append('"%s": "%s"' % (field, field))
        response_format = ', '.join(fields_json)
        prompt = template % (table_description, num_fields, fields)
        return prompt

class TableGenerator_JSON(TableGenerator):
    def parse_llm_response(self, response):
        tables = []
        md_answer, table_data = format_answer_text_with_table(response, [])

        # take the first table
        json_obj = table_data[0][1]
        headers = table_data[0][0]

        maximal_keyset = set()
        for obj in json_obj:
            maximal_keyset.update(obj.keys())

        assert len(maximal_keyset) == len(json_obj[0]), "All rows must have the same keys"

        return pd.DataFrame(json_obj, columns=headers)

class TableGenerator_MD(TableGenerator):
    def parse_llm_response(self, response):
        markdown_table_locations = find_markdown_tables(response)
        # take the first table
        table = markdown_table_locations[0].table
        return pd.DataFrame(table)
        

@dataclass
class ExperimentConfig:
    note: str
    template: str
    model: CompletionModel
    openai_api_key: str = os.getenv("OPENAI_API_KEY")
    openai_organization: str = os.getenv("OPENAI_ORGANIZATION")
    save_path: str = SAVE_PATH
    
def get_config(template: str, note: str, model: CompletionModel = CompletionModel.GPT_4O, save_path: str = SAVE_PATH) -> ExperimentConfig:
    return ExperimentConfig(
        note=note,
        template=template,
        model=model,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_organization=os.getenv("OPENAI_ORGANIZATION"),
        save_path=save_path
    )
    

class ExperimentRunner():
    
    def __init__(self, table_generator, metadata_path, config: ExperimentConfig):
        self.config = config
        with open(metadata_path, "rb") as f:
            self.metadata = json.load(f)
            
        self.table_generator = table_generator
        
        self.result_name = "%s_%s_%s" % (self.config.model.value.replace('-', '_'),                                                    
                                                   self.config.note,
                                                   time.strftime("%Y%m%d-%H%M%S"))
        
        self.result_folder = os.path.join(self.config.save_path, self.result_name)
        
        print("Experiment result folder: %s" % self.result_folder)
        
        os.makedirs(self.result_folder, exist_ok=True)
        os.makedirs("%s/tables" % self.result_folder, exist_ok=True)
        
        self.result = {}
        
    def fetch_data(self, idx: int):
        idx = str(idx)
        task = self.metadata[idx]
        
        task_name = task['name']        
        print("Fetching data for %s" % task_name)
        
        query, columns = task['table_title'], task['columns']   
        prompt = self.table_generator.generate_prompt(self.config.template, query, columns)        
        print(prompt)
        self.result[idx] = {'prompt': prompt}        
            
        try:
            max_tokens = calculate_max_output_tokens(prompt, '', model=self.config.model)
            client = openai.OpenAI(
                api_key=self.config.openai_api_key,
                organization=self.config.openai_organization,
            )
            result = client.chat.completions.create(
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                model=self.config.model.value,
                max_tokens=max_tokens,
            )

            response = result.choices[0].message.content.strip()
            
            if 'response' in self.result[idx]:
                self.result[idx]['response'].append(response)
            else:    
                self.result[idx]['response'] = [response]

            df = self.table_generator.parse_llm_response(response)

            df_ref = pd.read_csv(os.path.join(DATA_PATH, "tables", task['file']))        
            df.columns = df_ref.columns
            df = df.drop_duplicates(subset=task['keys'])

            table_path = "%s/tables/%s.csv" % (self.result_folder, task_name)
            self.result[idx]['table_path'] = table_path                
            df.to_csv(table_path, index=False)            

            print("Created table with %d rows" % len(df))

            return df

        except Exception as e:  
            print(e)
            print(e.__class__.__name__)
            
    def save_result(self):
        with open("%s/result.json" % self.result_folder, "w") as outfile:
            result_json = json.dumps(self.result, indent=4)
            outfile.write(result_json)

def main():
    console = setup_rich_logging(logging.INFO, log_to_file=True)
    runner_json = ExperimentRunner(TableGenerator_JSON(), 
                                  metadata_path=os.path.join(DATA_PATH, "cfg.json"), 
                                  config=get_config(TEMPLATE_FULL_TABLE_JSON, "full_table_json", CompletionModel.GPT_4O))
    runner_md = ExperimentRunner(TableGenerator_MD(), 
                                metadata_path=os.path.join(DATA_PATH, "cfg.json"), 
                                config=get_config(TEMPLATE_FULL_TABLE_MD, "full_table_md", CompletionModel.GPT_4O))

    with ThreadPoolExecutor(max_workers=25) as executor, Progress(console=console) as progress:
        futures = [executor.submit(runner_json.fetch_data, idx) for idx in range(100)]
        futures.extend([executor.submit(runner_md.fetch_data, idx) for idx in range(100)])

    for future in as_completed(futures):
        future.result()

    runner_json.save_result()   
    runner_md.save_result()   

if __name__ == "__main__":
    main()