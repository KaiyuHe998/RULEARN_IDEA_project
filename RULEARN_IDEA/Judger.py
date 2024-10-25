# basics This script is currently not in use
from __future__ import annotations
import re
from typing import *
import openai
import ast
import utils

# CHIBI components
from world_basic_blocks import Thing_container,Edge_Base

class Judger:
    Environment_description = None
    #Space_Manager_System_Global = Space_Manager_System_Global # system level space manager global
    Usage = None
    Model_name = 'gpt-3.5-turbo-0125'
    World_rules = None
    
    @classmethod
    def judge(cls,attemptation_action:'plan_system.Plan_attemptation_interactive_action')->Tuple[str,str]:
        assert attemptation_action.detailed_action_string is not None, f'''the detailed_action_string is None for the input action, need to call interact before judge!'''
        '''Need to return the interactive action result and the reason'''
        # all objects in related to this action
        related_objects = attemptation_action.get_relative_objects() 
        # step1:
        generated_action_result = cls._get_action_result_func(attemptation_action)()
        action_success_fail_state = generated_action_result['parsed_result'][0]
        action_success_fail_reason = generated_action_result['parsed_result'][1]
        assert isinstance(action_success_fail_state, bool), f'''The parsed result {action_success_fail_state} is not a boolean.'''
        return (action_success_fail_state, action_success_fail_reason) 
        
    @classmethod
    def _get_action_result_func(cls,
                           attemptation_action:'plan_system.Plan_attemptation_action_Base',
                           )->Tuple[str,str]:
        @utils.Prompt_constructor_for_system(cls.Model_name, 
                                             Usage = cls.Usage,
                                             parse_function_str = 'str_with_tuple')
        def _prompt_and_input():
            Action_host_CHIBI = attemptation_action.Host_CHIBI
            if cls.World_rules is None:
                rule_str = 'Common sense'
            else:
                rule_str = cls.World_rules
            related_objects = attemptation_action.get_relative_objects() 
            related_objects_str = ''
            cur_space_name = attemptation_action.Host_CHIBI.Space_manager.Cur_position
            for single_object in related_objects:
                related_objects_str += (single_object.get_keyword() + ':' + single_object.get_information(Action_host_CHIBI) + '\n')

            Prompt = f'''You are a system component tasked with evaluating the actions taken by {attemptation_action.Host_CHIBI.Name}. Initially, {attemptation_action.Host_CHIBI.Name} decides to {attemptation_action.get_information()}. Their specific course of action is detailed in {attemptation_action.detailed_action_string}. Additionally, there's detailed information about the object {attemptation_action.Host_CHIBI.Name} is currently interacting with: {related_objects_str}. Please follow these steps for your assessment:
Step 1: Analyze what is {attemptation_action.Host_CHIBI.Name}'s intention by doing this?

Step 2: Analyze each action detailed in the description, breaking it down into separate actions. For each action, identify the object involved and recall its detailed information, focusing on key properties relevant to the action, and make a short judegement of whether this object or action will function as {attemptation_action.Host_CHIBI.Name} intened. Your judgement should strictly align to physicial law and common sense. And your judgement should strictly follows to the object's detailed information.

Step 3: Summarize your conclusion in a tuple. The first element should indicate whether {attemptation_action.Host_CHIBI.Name}'s plan was successful or he fullfilled his intention (True or False). The second element should explain the reasons for the success or failure of the action.
'''
            Input = f'''Example Answers: \nexample1:
**Step 1 Intention:**
Michael's action indicating that he want to create a bullet with materials and tools he has.
**Step 2 Analysis:**
Measuring Gunpowder: While Michael's precision in measuring the gunpowder is commendable, the wet condition of the gunpowder impacts its effectiveness. The moisture causes clumping, making it challenging to achieve an accurate weight. Clumped gunpowder can lead to inconsistent burn rates, potentially affecting the bullet's performance and safety.
Filling Casing with Gunpowder: The transfer of the wet, clumped gunpowder into the bullet casing further complicates the process. Clumps can create uneven distribution within the casing, leading to potential firing issues or misfires.
Placing the Primer: Despite the condition of the gunpowder, placing the primer at the base of the casing is done correctly. However, the primer's effectiveness might be compromised if the moisture from the gunpowder affects it.
Securing the Bullet Head: Attaching the bullet head completes the assembly process, but the overall integrity of the bullet is questionable due to the wet gunpowder. The bullet might not perform as expected when fired.

**Step 3 Conclusion:**
Generated tuple: (False, "The effectiveness of the bullet is compromised due to the wet condition of the gunpowder. Moisture causes the gunpowder to clump, making precise measurement difficult and leading to potential inconsistencies in burn rates. These issues can result in uneven distribution within the casing, potential misfires, and a reduction in the overall safety and performance of the bullet.")


example2: 
**Step 1 Intention:**
Emily's action indicating that she want lock the door.
**Step 2 Analysis:**
Closing the Cabin Door: Emily's action of firmly grasping the cabin door handle and swinging it closed ensures that the door is tightly shut. This is an essential step in securing the cabin against external elements, such as wind and wildlife. A tightly shut door helps maintain warmth inside the cabin and provides a physical barrier against external threats.
Locking the Door: Turning the lock to secure the door further enhances the cabin's safety. This action not only prevents potential wildlife intrusions but also adds an additional layer of security against unwelcome human entry. The lock is a critical component in ensuring that the cabin remains a safe and warm haven in a potentially hostile environment.

**Step 3 Conclusion:**
Generated tuple: (True, "Emily's actions successfully secure the cabin for safety and warmth. By firmly closing and locking the door, she effectively shields the interior from the howling wind and potential wildlife intrusions, thus maintaining a safe and warm environment inside the cabin.")

Now please generate the result for {attemptation_action.Host_CHIBI.Name}, please carefully follow the sample answer and instructions, and the success or not should strongly related to the descrption of {attemptation_action.Interactive_object.get_keyword()}, please be objective and rigorous.'''
            return Prompt, Input
        return _prompt_and_input