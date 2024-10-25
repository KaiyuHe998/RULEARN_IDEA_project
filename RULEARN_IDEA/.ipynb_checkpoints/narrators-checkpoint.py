# basics
from abc import ABC, abstractmethod
from typing import *

# CHIBI components
import utils

class State_edit_narrator:
    '''This class is a narrator that focus on narrate the CHIBI's action trajectory and helps generated action are more consistent and logical'''
    narrative_styles = {'Third_person_realistic':'Please summarize character\'s structured path of action into a complete text description in a more fluid manner from a third-person perspective. First of all, please don\'t add any additional information to the input, don\'t add any details that aren\'t covered in the description. Secondly, do not lose any of the information mentioned in the input. Your description should be precise and coherent. Do not add any unstated results or predictions about the outcome of the operation.'}
    Model_name = 'gpt-3.5-turbo-0125'
    Usage = {'completion_tokens': 0,   # returned by parse function
             'prompt_tokens':0,            # openai usage
             'total_tokens': 0}
    @staticmethod
    def narrate(state:'Plan_system.State_Simple',
                narrative_style_key:str = 'Third_person_realistic',
                Model_name:Optional[str] = None):
        assert narrative_style_key in State_edit_narrator.narrative_styles, f'{narrative_style_key} is not defined.'
        narrative_style = State_edit_narrator.narrative_styles[narrative_style_key]

        if Model_name is None:
            Model_name = State_edit_narrator.Model_name
        @utils.Prompt_constructor_for_system(Model_name,
                                             Usage = State_edit_narrator.Usage,
                                             parse_function_str =None,)
        def _prmopt_input_for_narrative():

            Prompt = narrative_style
            Input = f'''Background Overview: {state.Information}. 
Action Summary: Following is the action that character took or the observations of the character:\n{state.Tried_action_str}. 
Guidelines: 
1. Accuracy is key. Ensure that the summary accurately reflects only the actions taken, without adding or omitting details.
2. Focus on coherence. Make sure the summary logically aligns with the background information, avoiding repetition.
3. Keep it concise. If the action and result are complex, distill them to their essence, akin to a diary entry, If the experience mentioned imporant clue or information, please clearly mention this clue in the summarized sentence.
4. Exclude extraneous details. Mention only those elements directly related to the actions taken.
Your task: Craft a summary that seamlessly integrates with the existing background, providing a clear and concise account of the latest actions. Please don't repeat what has already narrated in the background. If the clue and evidence is important please desribe it in detail and do not change the meaning of it from observation.'''
            return Prompt, Input
        generated_result = _prmopt_input_for_narrative()
        if State_edit_narrator.Usage is None:
            State_edit_narrator.Usage = generated_result['new_usage']
        return generated_result['parsed_result']


# class Environment_Describe_Narrator:
#     '''This class is a narrator that focus on provide overview information of a space to CHIBI'''
#     narrative_styles = {'''Normal''':'''You are a a'''}
#     Model_name = 'gpt-3.5-turbo-0125'
#     Usage = {'completion_tokens': 0,   # returned by parse function
#              'prompt_tokens':0,            # openai usage
#              'total_tokens': 0}
    