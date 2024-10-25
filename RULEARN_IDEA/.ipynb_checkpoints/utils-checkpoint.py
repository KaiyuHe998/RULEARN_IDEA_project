import re
import numpy as np
from typing import *
import openai
import ast
import logging
import threading
import datetime
import os
import time
import pandas as pd
import gc
import torch
import json
import traceback


# ---------------------------------------------------Basic utils---------------------------------------------------
def get_all_finished_experiments(csv_file_name:str)->List[Dict[str,Any]]:
    assert os.path.exists(csv_file_name), f'''csv_file_name:{csv_file_name} do not exist'''
    return_list = []
    cur_result_csv = pd.read_csv(csv_file_name)
    for index, row_data in cur_result_csv.iterrows():
        tem_puzzle_parameters = {'puzzle_name':row_data['puzzle_name'],
                                 'level':int(row_data['puzzle_level']),
                                 'puzzle_index':int(row_data['puzzle_index']),
                                 'Do_abduction':bool(row_data['Do_abduction']),
                                 'round_index':row_data['round_index'],
                                 'forced_abduction':row_data['forced_abduction']}

        return_list.append(tem_puzzle_parameters)
    return return_list

def calculate_cosine_similiarity(vector_1, vector_2):
    # Calculate the dot product of the two vectors
    dot_product = np.dot(vector_1, vector_2)

    # Calculate the magnitude of each vector
    magnitude_vector_1 = np.linalg.norm(vector_1)
    magnitude_vector_2 = np.linalg.norm(vector_2)

    # Calculate the cosine similarity
    if magnitude_vector_1 == 0 or magnitude_vector_2 == 0:
        # Avoid division by zero
        return 0
    else:
        cosine_similarity = dot_product / (magnitude_vector_1 * magnitude_vector_2)
        return cosine_similarity

def cancel_all_batch_task(client, log_file_path):
    all_api_batch_information = []
    with open(log_file_path, 'r') as file:
        for line in file:
            all_api_batch_information.append(line.strip()) # read every generated sentence
    for log_information in all_api_batch_information:
        match = re.search(r"id='([^']*)'", log_information)
        
        if match:
            batch_id = match.group(1)
            print("Found batch ID:", batch_id)
            try:
                client.batches.cancel(batch_id)
            except Exception as e:
                print(e)
            finally:
                batch_job = client.batches.retrieve(batch_id)
                print('------------------------------------')
                print(batch_job)
                print('------------------------------------')
        else:
            print("Batch ID not found.")
            
def is_convertible_to_int(s):
    try:
        int(s)  
        return True
    except ValueError:
        return False  

def setup_logging(log_filename):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if logger.handlers:
        for handler in logger.handlers:
            logger.removeHandler(handler)

    file_handler = logging.FileHandler(log_filename)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

def generate_logger(name, log_file, level=logging.INFO):
    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger

# ---------------------------------------------------Base Block utils---------------------------------------------------


# -------------------------------------------------Knowledge Base utils---------------------------------------------------


# -------------------------------------------------Space utils----------------------------------------------------------


# -------------------------------------------------Openapi api utils---------------------------------------------------

# Parse functions -------------------------      
def process_tuple_elements(input_str):
    # this function is to make sure that the generated str can be converted to tuple or other stuff as needed
    cleaned_input = input_str.strip('()')
    elements = cleaned_input.split(',')
    
    processed_elements = []
    for element in elements:
        element = element.strip()
        try:
            ast.literal_eval(element)
            if element != '':
                processed_elements.append(element)
        except:
            processed_elements.append(f"'{element}'")
    return f"({', '.join(processed_elements)})"

def _parse_str_with_tuple(generated_str):
    matches = re.findall(r'\(([^()]*)\)', generated_str)
    match = matches[-1].replace('\\','')
    try:
        match = process_tuple_elements(match)
        return ast.literal_eval(match)
    except:
        return match
        
def _parse_str_with_angle_bracket(generated_str):
    matches = re.findall(r'<([^<>]*)>', generated_str)
    match = matches[-1].replace('\\','')
    try:
        return ast.literal_eval(match)
    except:
        return match

def _parse_str_with_square_bracket(generated_str):
    matches = re.findall(r'\[([^[\]]*)\]', generated_str)
    match = matches[-1].replace('\\','')
    try:
        return ast.literal_eval(match)
    except:
        return match

class Prompt_batch_generator:
    # used for batch evaluation, run a batch of puzzles in parallel.
    def __init__(self, 
                 model_name = None,
                 model_path = "CHIBI_models", 
                 max_length = None,
                 token:Optional[str] = None, # hugging face token or openai token
                 batch_size:Optional[int] = None,
                 visible_gpu_list:Optional[List[int]] = None,
                 repeat_previous_experiment:bool = False,
                 openai_model_list:List[str] = []):
        assert batch_size is not None, f'''batch is None'''
        self.Model_name = model_name
        self.Model_path = model_path
        self.Tokenizer = None
        self.Model = None
        self.max_length = max_length
        self.token = token # could be huggingface token or openai api token
        self.visible_gpu_list = visible_gpu_list
        self.openai_model = openai_model_list# if your openai model is not listed here will goes to hugging face pipeline
        assert len(self.openai_model) >0 ,f'''You should pass this into the batch generator'''
        if repeat_previous_experiment:
            pass
        else:
            self.init_model()
        self.batch_size = batch_size
        self.batch_data = []
        self.condition = threading.Condition()
        self.lock = threading.Lock()
        self.total_batch_processed = 0
        self.average_batch_time = datetime.timedelta(seconds = 0)
        self.Task_start_time = datetime.datetime.now()
        self.all_experiment_parameters = None
        self.Total_finished_tasks = 0
        self.Total_failed_tasks = 0
        self.processed_data = {}
        if self.batch_size == 1: # Set batch_size for test use
            self.print_message = True
        else:
            self.print_message = False
        
    def register_task(self, task_parameter:Dict[str, Any]):
        with self.condition:
            assert task_parameter in self.all_experiment_parameters,f'''running parameter should in task list not finished'''
            self.all_experiment_parameters.remove(task_parameter)
            print(f'''now start a new task parameter: {task_parameter}''')
            print(f'''There are {len(self.all_experiment_parameters)} tasks left.''')
            
    def unregister_task(self, task_parameter:Dict[str,Any], puzzle_drop:bool):
        with self.condition:
            if len(self.all_experiment_parameters) == 0: #no tasks left
                self.batch_size -= 1
                if self.batch_size == len(self.batch_data):
                    if self.batch_size != 0:
                        current_thread = threading.current_thread()
                        print(f"Current thread name: {current_thread.name} finished total worker - 1 and, letting the rest workers generating.")
                        self.processed_data = self.generate()
                        print(f"Current thread name: {current_thread.name} batch data generated complete, waking up all threads")
                        self.batch_data = []
                        self.condition.notify_all()
                    else: # all tasks finished
                        print(f'All tasks finished')
                        assert self.batch_data == [], f'''All tasks finished batch_data should be empty, but currently there is data in it: batch_data:{self.batch_data}'''
            if not puzzle_drop:
                self.Total_finished_tasks += 1
            else:
                self.Total_failed_tasks += 1
                

    def add_data(self, input_data):
        with self.condition:
            current_thread = threading.current_thread()
            
            self.batch_data.append(input_data)
            if len(self.batch_data) < self.batch_size:
                print(f"Current thread name: {current_thread.name} data added and waiting for batch full.")
                print(f'''Current batch data number: {len(self.batch_data)}, Current_batch_size: {self.batch_size}''')
                self.condition.wait()
            else:
                print(f"Current thread name: {current_thread.name} data added and batch is full now start generating.")
                try:
                    self.processed_data = self.generate()
                    torch.cuda.empty_cache()
                    print(f"Current thread name: {current_thread.name} batch data geration success, waking up all threads")
                    self.batch_data = []
                    self.condition.notify_all()  # generate finished notify all thread to fetch the generated result
                except Exception as e:
                    torch.cuda.empty_cache()
                    print(f"Current thread name: {current_thread.name} batch data generated Failed, waking up all threads")
                    self.batch_data = []
                    self.condition.notify_all()  # generate finished notify all thread to fetch the generated result
                    print(f'''error occured, {e}''')
                    raise e


    def _generate_with_openai_api(self, input_data:List[str])->List[str]:
        tasks = []
        input_id_match_dict = {}
        
        for index, prompt_data in enumerate(input_data):
            custom_id = f'Task_{index}'
            input_id_match_dict.update({custom_id:prompt_data})
            task = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": self.Model_name,
                    "temperature": 0.3,
                    "response_format": { 
                        "type": "json_object"
                    },
                    "messages": [
                        {
                            "role": "system",
                            "content": '''You are a helpful assistant that generates content exactly following the user's instructions and format requirements (especially placing your final decision output in brackets or parentheses when required), when you making decisions or reasons please based on all your observations do not say someting contradicting to your hypothesis and observations, and you organize your output in JSON.''' # openai batch api needs to use JSON format, if not include this JSON they will not give your result correctly
                        },
                        {
                            "role": "user",
                            "content": prompt_data
                        }
                    ],
                }
            }
            tasks.append(task)
        file_name = f"{self.Model_name}_batch{self.total_batch_processed}.jsonl"
        submit_json_file_path = os.path.join(self.data_save_root_path,'submit')
        result_json_file_path = os.path.join(self.data_save_root_path,'result')
        if not os.path.exists(submit_json_file_path):
            os.makedirs(submit_json_file_path)
        if not os.path.exists(result_json_file_path):
            os.makedirs(result_json_file_path)
        save_file_path_submit_json_file = os.path.join(submit_json_file_path,file_name)
        save_file_path_result_json_file = os.path.join(result_json_file_path,file_name)
        
        try:
            cur_batch_start_time = datetime.datetime.now()
            with open(save_file_path_submit_json_file, 'w') as file:
                for obj in tasks:
                    file.write(json.dumps(obj) + '\n')
            with open(save_file_path_submit_json_file, "rb") as file:
                batch_file = self.client.files.create( 
                                                  file = file,
                                                  purpose="batch"
                                                )
            print(f'''Batch file created: {batch_file}''')
            batch_job = self.client.batches.create(
                  input_file_id=batch_file.id,
                  endpoint="/v1/chat/completions",
                  completion_window="24h"
                )
            print(f'''Batch job submitted: {batch_job}''')
            self.Logger.info(f'''Batch job submitted: {batch_job}''')

            while batch_job.status != 'completed':
                batch_job = self.client.batches.retrieve(batch_job.id)
                time.sleep(10) # sleep for 10 secs and check the state of the batch job
                openai_batch_end_time = datetime.datetime.now()
                cur_end_time = datetime.datetime.now()
                cur_time_used = cur_end_time - cur_batch_start_time
                print(f'''---Time used: {cur_time_used}, cur_status: {batch_job.status}---''', end = '\r')
            assert batch_job.output_file_id is not None, f'''Task completed but the no outputfile id'''
            #print('\nCur batch finished!')
            cur_batch_end_time = datetime.datetime.now()
            time_used = cur_batch_end_time - cur_batch_start_time
            self.total_batch_processed += 1
            self.average_batch_time = (self.average_batch_time*(self.total_batch_processed-1)+time_used)/self.total_batch_processed
            print(batch_job)
            batch_result = self.client.files.content(batch_job.output_file_id).content
            with open(save_file_path_result_json_file, 'wb') as file:
                file.write(batch_result)
            results = []
            with open(save_file_path_result_json_file, 'r') as file:
                for line in file:
                    # Parsing the JSON string into a dict and appending to the list of results
                    json_object = json.loads(line.strip())
                    results.append(json_object)
            return_result_dict = {}
            for res in results:
                custom_id = res['custom_id']
                result = res['response']['body']['choices'][0]['message']['content'][2:-2]
                return_result_dict.update({input_id_match_dict[custom_id]:result})
            return return_result_dict
        except Exception as e:
            print("An error occurred when generating with openai api call: ", e)
            traceback.print_exc()
            raise e
        finally:
            try:
                if batch_job.status != 'completed':
                    self.client.batches.cancel(batch_job.id)
            except Exception as e:
                print("An error occurred when cancelling the a batch: ", e)
                traceback.print_exc()
            
    def generate_with_openai_api(self):
        self.processed_data = {}
        max_attempts = 1
        attempt = 0
        while attempt <= max_attempts:
            try:
                print(f'''------------Start generating batch: {self.total_batch_processed}, Batch_size:{self.batch_size}, Average_batch_time:{self.average_batch_time}, Cur_time: {datetime.datetime.now()}, Tasks starts from: {self.Task_start_time}, Finished num tasks:{self.Total_finished_tasks}, Failed num tasks: {self.Total_failed_tasks}, Number of tasks remaining: {len(self.all_experiment_parameters)}----------------''')
                return self._generate_with_openai_api(self.batch_data)
            except Exception as e:
                max_attempts += 1
                print("An error occurred when generating with openai api call: ", e)
                traceback.print_exc()
                raise e

    
    def _generate_with_hg_model(self, input_data):
        input_ids = self.Tokenizer(input_data, return_tensors="pt", padding=True).to("cuda")
        outputs = self.Model.generate(**input_ids,
                                      max_new_tokens=self.max_length,     
                                      num_beams=1,                
                                      temperature=0.3,            
                                      top_p=None,                 
                                      top_k=None)
        generated_str = [self.Tokenizer.decode(outputs[i],skip_special_tokens=True) for i in range(len(outputs))]
        new_generated_outputs = []
        #assert len(generated_str) == len(input_data), f'''generated sequence length is not the same as input'''
        for item1, item2 in zip(self.batch_data, generated_str):
            new_content = item2[len(item1):]
            if 'sistant' in new_content:
                new_content = '\n'.join(new_content.split('\n')[1:])
            new_generated_outputs.append(new_content)
        return new_generated_outputs
        
    def generate_with_hg_model(self):
        self.processed_data = {}
        max_attempts = 1
        attempt = 0
        while attempt<=max_attempts:
            try:
                print(f'''------------Start generating batch: {self.total_batch_processed}, Batch_size:{self.batch_size}, Average_batch_time:{self.average_batch_time}, Cur_time: {datetime.datetime.now()}, Tasks starts from: {self.Task_start_time}, Finished num tasks:{self.Total_finished_tasks}, Failed num tasks:{self.Total_failed_tasks}, Number of tasks remaining: {len(self.all_experiment_parameters)}----------------''')
                start_time = datetime.datetime.now()
                new_generated_outputs = self._generate_with_hg_model(self.batch_data)
                end_time = datetime.datetime.now()
                time_used = end_time - start_time
                self.total_batch_processed += 1
                self.average_batch_time = (self.average_batch_time*(self.total_batch_processed-1)+time_used)/self.total_batch_processed
                torch.cuda.empty_cache()
                return {item[0]:item[1] for item in zip(self.batch_data,new_generated_outputs)}
            except Exception as e:
                attempt += 1
                print("An error occurred when generating : ", e)
                traceback.print_exc()
                if "CUDA out of memory" in str(e):
                    try:
                        first_cut_off = int(self.batch_size/3)
                        second_cut_off = int(2*(self.batch_size/3))
                        gc.collect()
                        torch.cuda.empty_cache()
                        start_time = datetime.datetime.now()
                        new_generated_outputs = []

                        
                        print(f"CUDA out of memory error encountered. Trying smaller batches(3 split). Now generating first part of the batch, batch size: {int(self.batch_size/3)}")
                        first_half_batch_generated_result = self._generate_with_hg_model(self.batch_data[:first_cut_off])
                        new_generated_outputs.extend(first_half_batch_generated_result)
                        del first_half_batch_generated_result
                        gc.collect()
                        torch.cuda.empty_cache()
                        
                        print(f"CUDA out of memory error encountered. Trying smaller batches(3 split). Now generating second part of the batch, batch size: {int(self.batch_size/3)}")
                        second_half_batch_generated_result = self._generate_with_hg_model(self.batch_data[first_cut_off:second_cut_off])
                        new_generated_outputs.extend(second_half_batch_generated_result)
                        del second_half_batch_generated_result
                        gc.collect()
                        torch.cuda.empty_cache()

                        print(f"CUDA out of memory error encountered. Trying smaller batches(3 split). Now generating third part of the batch, batch size: {int(self.batch_size/3)}")
                        third_half_batch_generated_result = self._generate_with_hg_model(self.batch_data[second_cut_off:])
                        new_generated_outputs.extend(third_half_batch_generated_result)
                        del third_half_batch_generated_result
                        gc.collect()
                        torch.cuda.empty_cache()
                        
                        end_time = datetime.datetime.now()
                        time_used = end_time - start_time
                        self.total_batch_processed += 1
                        self.average_batch_time = (self.average_batch_time*(self.total_batch_processed-1)+time_used)/self.total_batch_processed
                        torch.cuda.empty_cache()
                        return {item[0]:item[1] for item in zip(self.batch_data,new_generated_outputs)}
                    except Exception as e:
                        attempt += 1
                        print("An error occurred when generating with splitting the batch: ", e)
                        torch.cuda.empty_cache()
                        traceback.print_exc()
                        raise Exception("error occured when trying with split batch")
                        print("rest for 10 seconds and start retrying")
                else:
                    torch.cuda.empty_cache()
                    traceback.print_exc()
                    time.sleep(10)
                    raise Exception(f"An unexcepted error happens when genearting with {self.Model_name}")
                    
    def get_result(self, input_data):
        with self.condition:
            return_str = None
            if len(self.processed_data) == 0:
                return_str = "Generating Failed!!!!!"
            else:
                return_data = self.processed_data[input_data]
                #del self.processed_data[input_data]
                return_str = return_data
            return return_str
                
    def init_model(self):
        if self.Model_name not in self.openai_model: # using huggingface model
            from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
            import torch
            assert self.Model_name is not None, f'''Model name is None'''
            assert self.Model_path is not None, f'''Model path is None'''
            assert self.max_length is not None, f'''Model max_length is None'''
            print(self.Model_path)
            tokenizer = AutoTokenizer.from_pretrained(self.Model_name, token = self.token)
            model = AutoModelForCausalLM.from_pretrained(self.Model_name, device_map="auto", token = self.token, torch_dtype= torch.bfloat16).eval()
            model = torch.compile(model)
            self.Model = model
            self.Tokenizer = tokenizer
            self.Tokenizer.pad_token = self.Tokenizer.eos_token
            self.generate = self.generate_with_hg_model
            
        else: # using openai model
            self.generate = self.generate_with_openai_api
            self.data_save_root_path = f'''{self.Model_name}_api_data_folder'''
            if not os.path.exists(self.data_save_root_path):
                os.makedirs(self.data_save_root_path)
            self.client = openai.OpenAI(api_key=self.token)
            openai.api_key = self.token

            # create a log to keep track of all tasks submitted to openai
            logger_name = f'''openai_api_task_logger'''
            logger_file = os.path.join(self.data_save_root_path,logger_name)
            self.Logger = generate_logger(logger_name,logger_file)


# Prompt functions ------------------------
def Prompt_constructor_for_system(Model_name:str, 
                                  parse_function_str:Optional[str] = 'ast',
                                  Usage:Optional[Dict[str,int]] = None,
                                  print_generated_str:bool = False,
                                  print_prompt_and_input:bool = False,
                                  generative_mode:str = "Default",
                                  temperature:float = 0.3,
                                  logging_label:Optional[str] = None,
                                 ) -> Callable[...,Dict[str,Any]]:
    '''Prompt_constructor generate the prompt, decorator function helps generate/parse/use generated result'''
    def decorator(Prompt_constructor:Callable[Any,List[str]]):
        def wrapper(*args, **kwargs):
            return_stuff = Prompt_constructor(*args, **kwargs)
            Prompt = return_stuff[0]
            Input = return_stuff[1]
            if print_prompt_and_input:
                #print('-----------------------------------Prompt and Input--------------------------------------')
                print(f'Prompt: {Prompt}')
                print(f'Input: {Input}')
                
            if generative_mode == 'Default':
                generated_str, new_usage, messages = generate_with_prompt_and_input(Model_name, Prompt, Input,temperature = temperature)
            elif generative_mode == "Generative_session":
                generated_str, new_usage, messages = generative_session_with_prompt_and_input(Model_name, Prompt, Input, temperature = temperature)
                # TODO: need to store these generated messages
                print("TODO: need to store these generated messages")
            else:
                assert False, 'Unknown generative mode'
            #print(messages)
            #print(f'--------------------------Currently generating with{Model_name}----------------------------')
            if not Usage is None:
                add_token_usage(Usage,new_usage)
            if print_generated_str:
                print(generated_str)
                
            if parse_function_str is None:
                parsed_result = generated_str
            else:
                if parse_function_str == 'ast':
                    parse_function = ast.literal_eval
                elif parse_function_str == 'str_with_tuple':
                    parse_function = _parse_str_with_tuple
                elif parse_function_str == 'str_with_angle_bracket':
                    parse_function = _parse_str_with_angle_bracket
                elif parse_function_str == 'str_with_square_bracket':
                    parse_function = _parse_str_with_square_bracket
                else:
                    assert False, f'{parse_function_str} not known parse function'
                parsed_result = parse_function(generated_str)
            if logging_label is not None:
                Prompt_str = f'''**{logging_label}**: {Prompt}\n{Input}\n\n''' 
                generated_str = f'''**Generated_answer**: {generated_str}'''
                logging_information = Prompt_str+generated_str
                clean_logging_information = logging_information.replace('\n','<New Row>')
                logging.info(clean_logging_information)
                
            return {'parsed_result': parsed_result,   # returned by parse function
                    'new_usage':new_usage,            # openai usage
                    'messages': messages}             # List[str]
        return wrapper
    return decorator

def generate_with_prompt_and_input(Model_name:str, 
                                   Prompt:str, 
                                   Input:str,
                                   temperature:float = 0.3):
        messages = [
                {"role": "system", "content":Prompt},
                {"role": "user", "content": Input},
            ]
        generated_summary = openai.chat.completions.create(
          model=Model_name,
          messages=messages,
          temperature = temperature,
        )
        respond = generated_summary.choices[0].message.content
        
        messages.append({"role": "assistant", "content": generated_summary.choices[0].message.content})
        usage = {'completion_tokens':generated_summary.usage.completion_tokens,
                 'prompt_tokens':generated_summary.usage.prompt_tokens,
                 'total_tokens':generated_summary.usage.total_tokens}
        return respond, usage, messages


def generative_session_with_prompt_and_input(Model_name:str,  
                                             Init_prompt:str, 
                                             Init_input:str, 
                                             stop_sign:str = "End", 
                                             feed_back_mode:str = 'Default',
                                             temperature:float = 0.3):
    messages = [
            {"role": "system", "content":Init_prompt},
            {"role": "user", "content": Init_input},
        ]
    generated_summary = openai.chat.completions.create(
      model=Model_name,
      messages=messages,
      temperature = temperature)
    total_usage = {'completion_tokens':generated_summary.usage.completion_tokens,
                   'prompt_tokens':generated_summary.usage.prompt_tokens,
                   'total_tokens':generated_summary.usage.total_tokens}

    respond = generated_summary.choices[0].message.content
    
    if feed_back_mode == 'Default':
        feedback = input(f'{respond}\nWhat is your response？Input\'{stop_sign}\'to end the conversation')
    else:
        assert False, 'Unknown feedback mode'

    while feedback != stop_sign:
        messages.append({"role":"assistant", "content":respond})
        messages.append({"role":"user", "content":feedback})
        respond = generated_summary.choices[0].message.content
        cur_generated_summary = openai.chat.completions.create(
          model=Model_name,
          messages=messages)
        cur_usage = cur_generated_summary.usage
        cur_usage = {'completion_tokens':cur_usage.completion_tokens,
                   'prompt_tokens':cur_usage.prompt_tokens,
                   'total_tokens':cur_usage.total_tokens}
        respond = cur_generated_summary.choices[0].message.content
        add_token_usage(previous_usage=total_usage,
                              new_usage=cur_usage)
        feedback = input(f'{respond}\nWhat is your response？Input\'{stop_sign}\' to end the conversation')
        
    return respond, total_usage, messages

# -------------------------------------------------embedding-----------------------------------------------
def get_embedding(text:str, 
                  model:str="text-embedding-3-large"):
    # TODO need to save the data and embeddings when running
    # seems openai have a batch limitation so we just get one embeddings each time
    attempt_limit = 10
    cur_attempt = 0
    while cur_attempt<=attempt_limit:
        try:
            text = text.replace("\n", " ")
            if not text:
                assert False, 'input text for embedding is empty!!'
            embedding_respond = openai.embeddings.create(
                  input=[text], model=model)
            embedding = embedding_respond.data[0].embedding
            embedding_usage = embedding_respond.usage
            embedding_usage = {'prompt_tokens':embedding_usage.prompt_tokens,
                               'total_tokens':embedding_usage.total_tokens}
            return (embedding, embedding_usage, text)
        except Exception as e:
            cur_attempt += 1
            print("An error occurred when generating : ", e)
            if cur_attempt == attempt_limit:
                raise Exception("Max retry attempts reached, failing embedding openai api's fault.")
            print("rest for 30 seconds and start retrying")
            time.sleep(30)

#  -------------------------------------------------function call-----------------------------------------------
# currently this function is not in use 
def generate_with_function_call(Model_name:str, 
                                Input:str, 
                                functions:str):
    #latest models are gpt-3.5-turbo-1106 and gpt-4-1106-preview
    if len(functions) == 1: # only one function is provided
        function_name = functions[0]['name']
        generated_summary = openai.chat.completions.create(
                model=Model_name,
                messages=[{"role": "user", "content": Input}],
                functions=functions,
                function_call={"name": function_name},  # force to call this function, function_call='auto' is difault setting
            )
    else:
        assert False, 'Current can only input one functions a time'
    return (json.loads(generated_summary['choices'][0]['message']['function_call']['arguments']),generated_summary.usage)


def add_token_usage(previous_usage:Optional[Union[Dict[str,int],'openai.openai_object.OpenAIObject']] = None,
                    new_usage:Optional[Union[Dict[str,int],'openai.openai_object.OpenAIObject']] = None):
    if len(previous_usage) == 2: # currently, only embedding usage have two pairs
        previous_usage['prompt_tokens'] += new_usage['prompt_tokens']
        previous_usage['total_tokens'] += new_usage['total_tokens']
    else:
        previous_usage['completion_tokens'] += new_usage['completion_tokens']
        previous_usage['prompt_tokens'] += new_usage['prompt_tokens']
        previous_usage['total_tokens'] += new_usage['total_tokens']
    
def content_after_special_token_parse_function_constructor(special_token:str)->Callable[str,str]:
    '''retrun a parse function that match all content after defined special token'''
    def content_after_special_token_parse_function(input_str:str)->str:
        regex_pattern = rf'(?<={special_token}).+'

        match = re.search(regex_pattern, input_str, re.DOTALL)  
        if match:
            return match.group()
        else:
            assert False, 'Match pattern not found'
    return content_after_special_token_parse_function

def embedding_test(key_list:List[str],
                   query_list:List[str]):
    key_embeddings = [get_embedding(i)[0] for i in key_list]
    query_embeddings = [get_embedding(i)[0] for i in query_list]
    for i in range(len(key_list)):
        for j in range(len(query_list)):
            print(f'''{key_list[i]} and {query_list[j]} cos similarity: {calculate_cosine_similiarity(key_embeddings[i],query_embeddings[j])}''')
            
def decorate_text_with_color(input_text: str, color: str, deep: bool = False, bold: bool = False):
    color_dict = {
        'blue': '4',
        'red': '1',
        'yellow': '3',
        'green': '2',
        'purple': '5',
        'cyan': '6',      
        'magenta': '5',   
        'black': '0',
    }

    color_code = color_dict[color]
    color_code = f'9{color_code}' if not deep else f'3{color_code}'
    bold_code = '1' if bold else '22'

    return f'\033[{bold_code};{color_code}m' + input_text + '\033[0m'


class TaskCompletedException(Exception):
    """Exception raised when the agent completes its task."""
    pass

class TaskFailedException(Exception):
    """Exception raised when the agent completes its task."""
    pass

class GenerateErrorException(Exception):
    """generate function did not generate result correctly"""
    pass
