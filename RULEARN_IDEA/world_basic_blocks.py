# basics
from typing import *
from abc import ABC, abstractmethod
import utils
import re
import datetime
import math

class CHIBI_Object(ABC):
    '''In a world purely rely on Strings, 
    this is a basic block to build the whole world,
    a basic block should have the following functions and attributes'''
    def __init__(self,
                 Keyword:str,
                 Information:str,
                 Editable:bool = True,
                 Keyword_editable:bool = True,
                 Usage:Optional[Dict[str,int]] = None,
                 Model_name:str = 'gpt-3.5-turbo-0125',#gpt-4-1106-preview,
                 Belongs_to:Optional['Object'] = None,
                ):
        self.Keyword = Keyword
        self.Information = Information
        self.Editable = Editable
        self.Model_name = Model_name
        self.Usage = Usage
        self.Keyword_editable = Keyword_editable
        self.Belongs_to = Belongs_to
        if Usage is None:
            self.Usage = {'completion_tokens': 0,   # returned by parse function
                          'prompt_tokens':0,            # openai usage
                          'total_tokens': 0} 
        else:
            self.Usage = Usage
        # Keyword should be a brief identical summary (with in 5 words) of Information so that can be easily classified and identitfied。
    @abstractmethod
    def show(self):
        '''different type of object can have different show method directly print out all information in this object'''
        pass
    
    @abstractmethod
    def get_information(self,viewer:Optional[Union['system','CHIBI.CHIBI_Base']] = None) -> str:
        '''Viewer can only be 'system' or CHIBI class. If viewer is CHIBI sub class should return information needed for the CHIBI, if viewer is system, should return ground truth value'''
        return self.Information
    
    @abstractmethod
    def get_keyword(self):
        pass

    @abstractmethod
    def destory(self):
        '''some thing happens when this thing is deleted'''
        # Pointers need to care about
        # 1, Space Things
        # 2, Container
        # 3, 
        if self.Belongs_to is not None:
            self.Belongs_to.object_delete(self)
        # find the object in the space? and loop and delete?

        # if the object is also in a Space need to delete this form that space
        # impression de delete

    @abstractmethod    
    def edit(self,
             edited_information:str):
        '''Only edit information based on the action string, in the aspect of the object, what will happen after the action'''
        if self.Editable:
            self.Information = edited_information
        else:
            assert self.Editable, 'This Object is not editable, this function shouldn\'t be called'

    @abstractmethod
    def interact_pipeline(self, attemptation_action:'plan_system.Attemptation_Interactive_Action'):
        '''This function should handle all the effect (both semantic and sysmatic) after a interactive action taken place'''
        # First do semantic parse
        still_exist = self.semantic_parse(attemptation_action)

        # Second do systemic parse
        self.systemic_parse(attemptation_action)

        # Finally do some thing else
        if not still_exist:
            self.destory()

    def systemic_parse(self, attemptation_action:'plan_system.Attemptation_Interactive_Action'):
        '''some systemic level information should be changed in this step, eg object type, CHIBI open the box so the contained stuff should be exposed to every body, if CHIBI pass through the edge, CHIBI should move to the other side'''
        '''Can only be called if the action is success by judger'''
        '''Each object should have different systemic_parse pipeline'''
        pass

    def semantic_parse(self,
                       attemptation_action:'plan_system.Attemptation_Interactive_Action')->bool:
        '''This function should do all sorts of semantic parse including the information edit and some residuals remains after the interaction'''
        do_semantic_parse = self._get_semantic_parse_decision_func(attemptation_action)()
        if do_semantic_parse['parsed_result']:
            # step1 notify the change to related objects
            object_change_description = self._get_semantic_parse_func(attemptation_action)()['parsed_result']
            object_change_experience = f'''After you interacted with {self.get_keyword()}, You see {object_change_description}'''
            semantic_parse_experience = ''
            semantic_parse_experience += object_change_description
            

            # detect any subparts are created
            sub_parts_result = self._infer_generated_parts(attemptation_action,object_change_description)()['parsed_result']
            if isinstance(sub_parts_result, bool):
                sub_parts_result = (sub_parts_result,)
            if sub_parts_result[0]:
                for sub_part_name in sub_parts_result[1:]:
                    sub_part_description = self._get_generated_subparts_information(attemptation_action, object_change_description, sub_part_name)()['parsed_result']
                    new_item = Thing(sub_part_name, sub_part_description, Model_name = self.Model_name, Belongs_to = self.Belongs_to)
                notice_new_object_string = f'''After your action you noticed there are {', '.join(sub_parts_result[1:])} created during the process.'''
                semantic_parse_experience += notice_new_object_string

            still_exist = self._object_destory_decision(attemptation_action, object_change_description)()['parsed_result']
            if not still_exist:
                semantic_parse_experience += f'''The {self.get_keyword()} was rendered non-existent following the interaction.'''
            else: # the object is not destoried and should change its description
                new_name, new_information = self._get_new_information(attemptation_action, object_change_description)()['parsed_result']
                impression_object = [impression_object for impression_object in attemptation_action.Host_CHIBI.Space_manager.get_cur_space(space_type = 'impression').All_objects if impression_object.Impression_of is self][0]
                impression_object.Keyword = new_name
                self.Keyword = new_name
                self.edit(new_information)

            attemptation_action.Host_CHIBI.Memory_stream.memory_add(semantic_parse_experience)
            
            return still_exist
                
        else: # the object basically not change after the interaction given the detailed action string
            return False # no change so not destoried
        
    def _get_semantic_parse_decision_func(self, attemptation_action:'plan_system.Attemptation_Interactive_Action',)->Callable[Any,bool]:
        '''Decide if the state, function of the object have a notable change, should return true and Flase'''
        @utils.Prompt_constructor_for_system(self.Model_name,
                                             Usage = self.Usage,
                                             parse_function_str = 'str_with_tuple')
        def _prompt_and_input():
            Prompt = f'''You are a rule component within a system. You need to determine if {self.get_keyword()} with its detailed information:\n{self.get_information('system')} have significant change after an interaction. Below is a detailed description of this interaction:{attemptation_action.detailed_action_string}.\nAnd the result of the above action: \n{attemptation_action.Success_fail_reason}\n\nBased on the laws of physics and general common sense, please determine whether there have been any noticeable changes to {self.get_keyword()}'s physical properties, shape, appearance, functionality, integrity, or state as a result of the interaction. For example, a noticeable change can include a bent iron rod after being waved, a door changing from locked to open, or a glass of water being half-drunk. However, minor scratches or slight bumps that leave negligible marks are not significant changes. Please give your answer based on the following steps.'''
            Input = f"**Step1**:Analysis the action described interacting with {self.get_keyword()} is there sign indicating the  physical properties, shape, appearance, functionality, integrity, or state has significantly changed?\n\n**Step2**:Generate your result:Your response should be in the form of a tuple with a single element, indicating 'True' or 'False'. For example, answer (True) if you believe there have been significant changes to {self.get_keyword()}'s physical properties, shape, appearance, functionality, integrity, or state. Answer (False) if the object do not have significant change."
            return Prompt, Input
        return _prompt_and_input

    def _get_semantic_parse_func(self, 
                                attemptation_action:'plan_system.Attemptation_Interactive_Action'):
        @utils.Prompt_constructor_for_system(self.Model_name,
                                             Usage = self.Usage,
                                             parse_function_str = 'str_with_tuple')
        def _prompt_and_input():
            Prompt = f'''You are a rule component within a system. Your task is to narrate whether an object has undergone any changes in the following attributes as a result of a specific action interaction: physical properties, appearance, functionality, and state. Please describe the changes in detail, considering the main characteristics of the object and how it was interacted with in the action description. Combine the description of the object and the specific manner of interaction to meticulously explain the changes based on physical laws and common sense.'''
            Input =  f'''Object name: {self.get_keyword()}.\nObject detailed information: {self.get_information('system')}.\nDetailed interactive action description: {attemptation_action.detailed_action_string}.\nYou are required to describe, in one detailed sentence, how {self.get_keyword()} will be altered as a result of the interaction. This sentence should be enclosed in parentheses. For instance: ('After the interaction, the {self.get_keyword()} will exhibit changes such as...'). Your description must adhere strictly to the given facts, and all deductions should be based on common sense and physical laws. The change description should focus solely on the factual outcome of the interaction, avoiding any subjective interpretation of the action.'''
            return Prompt, Input
        return _prompt_and_input
    
    def _infer_generated_parts(self, 
                               attemptation_action:'plan_system.Attemptation_Interactive_Action',
                               object_change_description:str):
        @utils.Prompt_constructor_for_system(self.Model_name,
                                             Usage = self.Usage,
                                             parse_function_str = 'str_with_tuple')
        def _prompt_and_input():
            Prompt = f'''As a rule component in a system, you are faced with a situation involving a {self.get_keyword()} that has undergone a specific interaction, described as follows: {attemptation_action.detailed_action_string}.\n And the result of the above action:\n{attemptation_action.Success_fail_reason}. Additionally, there is a change description for the object during the interaction: {object_change_description}. Your task is to analyze these changes and determine if any new, DISTINCT, INDEPENDENT subparts of the {self.get_keyword()} have been created as a result. For example this could be a lock after the traditional chest box with a lock is opened, or piece of brick if the wall is destoryed.'''
            Input = f'''The detailed information of the {self.get_keyword()} is as follows: {self.get_information('system')}, please give your answer based on the following steps.\n**Step1** Analyze the detailed information of {self.get_keyword()}, is there mentioned any independent subparts mentioned? If there are no clear component or subparts mentioned in the detailed information of {self.get_keyword()} you should answer (False) directly as final answer.\n**Step2** Based on the description of the action, is there any sign indicating the subparts you found in Step1 break off and form a new independent subpart? If there are no subparts founded in Step1, you should not mention any subparts in this step.\n**Finally** Compile your findings in a tuple. The first element of the tuple should be 'True' or 'False', indicating your conclusion about the creation of subparts. If there are subparts, follow this with the names of these subparts. For example, your answer could be either (False) or (True, "Subpart_Name_1", "Subpart_Name_2"......) the numbers should rely on your analysis in Step2. If there are not Subparts mentioned in Step1, you should answer (False)'''
            return Prompt, Input
        return _prompt_and_input
        
    def _get_generated_subparts_information(self,
                                            attemptation_action:'plan_system.Attemptation_Interactive_Action',
                                            object_change_description:str,
                                            subpart_name:str):
        @utils.Prompt_constructor_for_system(self.Model_name,
                                             Usage = self.Usage,
                                             parse_function_str = 'str_with_tuple')
        def _prompt_and_input():
            Prompt = f'''As a rule component within a system, your task is to craft a detailed description of a subpart, named {subpart_name}, that originated from a {self.get_keyword()}. This subpart was created as a result of an interaction, detailed as: {attemptation_action.detailed_action_string}. The object {self.get_keyword()} has specific characteristics, which are: {self.get_information('system')}. Your description should closely follow the style, information, and facts provided in these descriptions.'''
            Input = f'''To generate the description, proceed with the following steps:
Step 1: Contemplate the shape, material, and other functional and notable details of {subpart_name}, considering its origin and the manner in which it was created.
Step 2: Adopt a third-person perspective. Focus solely on the current state of {subpart_name}, disregarding any actions or events that have already occurred. Describe the subpart based on the details you identified in Step 1.
Final Step: Summarize your description in a single-element tuple. For example, a lock from a locked chest could have the description as follows: ('This lock, crafted from high-grade steel, once secured a vintage oak chest. Its compact, cylindrical design, typical of early 20th-century craftsmanship, features a complex pin tumbler mechanism. Although structurally sound, the lock exhibits clear signs of tampering: the keyhole is distorted, likely from picking attempts, and the shackle shows stress marks and a slight twist, indicative of the force applied to pry it open.'). Please be noticed that any fact and statement in the generated description should strictly rely on given fact, please don't fake any detailes, if there are not enough facts, just make the description short and accurate.'''
            return Prompt, Input
        return _prompt_and_input

    def _object_destory_decision(self,
                                 attemptation_action:'plan_system.Attemptation_Interactive_Action',
                                 object_change_description:str):
        @utils.Prompt_constructor_for_system(self.Model_name,
                                             Usage = self.Usage,
                                             parse_function_str = 'str_with_tuple')
        def _prompt_and_input():
            Prompt = f'''In your role as a rule component within a system, you are tasked with determining whether an object exists after undergoing a specific interaction. The interaction in question is detailed as: {attemptation_action.detailed_action_string}. Additionally, there is a description of how the object has changed: {object_change_description}. By 'exists' we mean that the object should, in some form, still be recognizable as {self.get_keyword()}, regardless of whether it still functions as intended. If the object is entirely consumed (leaving nothing behind, not even a container) or if it is broken into unrecognizable pieces, it is considered to no longer exist.'''
            Input = f'''Based on the detailed description of the interaction and the changes it caused, determine if the object still exists. Summarize your decision in a tuple containing only one boolean value. For example, respond with (True) if you believe the object still exists, or (False) if you believe it no longer exists.'''
            return Prompt, Input
        return _prompt_and_input

    def _get_new_information(self,
                             attemptation_action:'plan_system.Attemptation_Interactive_Action',
                             object_change_description:str):
        @utils.Prompt_constructor_for_system(self.Model_name,
                                             Usage = self.Usage,
                                             parse_function_str = 'str_with_tuple')
        def _prompt_and_input():
            Prompt = f'''In your role as a rule component within a system, you are charged with the task of revising the name and description of an object following a specific interaction. The details of the interaction are provided as: {attemptation_action.detailed_action_string}. Furthermore, the changes the object has undergone are described in: {object_change_description}. Your objective is to modify the object's description based on both its original attributes and the changes it has experienced. Retain the aspects of the original description that remain unchanged. The original description of {self.get_keyword()} before the changes is: {self.get_information('system')}. Consider if a new name is warranted based on these modifications.'''
            Input = f'''Please generated your answer based on the following steps.\n**Step1** Analyze the change description to determine which parts of the original description should be updated and which should remain as they are, decide if there any new description needed for noticable change after the action.\n**Step2** Based on your analysis, create a new description. This should incorporate the changes while retaining unchanged elements.\n**Step3**  Evaluate whether the original name {self.get_keyword()} still fits the updated description.\n**Finally** Compile your findings in a tuple with two elements. The first element is the new or retained name of the object. The second element is the updated description. For instance: ("Iron Lock", "This lock, crafted from high-grade steel, once secured a vintage oak chest. Its compact, cylindrical design, typical of early 20th-century craftsmanship, features a complex pin tumbler mechanism. The lock shows signs of tampering: the keyhole is distorted, and the shackle is stressed and twisted."). Ensure that every detail in your new description strictly aligns with the provided facts. If certain details are missing, focus on making the description brief yet precise.'''
            return Prompt, Input
        return _prompt_and_input
        
    
# -------------------------------Thing related classes------------------------------------
# -------------------------------Thing related classes------------------------------------
# -------------------------------Thing related classes------------------------------------
class Thing(CHIBI_Object):
    '''This is a stuff that can be seen in a world,
    not pure information'''
    def __init__(self,
                 Keyword:str,
                 Information:str,
                 Editable:bool = True,
                 Keyword_editable:bool = True,
                 Model_name:str = 'gpt-3.5-turbo-0125',
                 Usage:'openai.openai_object.OpenAIObject' = None,
                 Belongs_to:Union["Thing_container",'space_manager.Space_Base'] = None,
                 #Relational_memory:Optional["Experience_container"] = None,
                 #Interactive_actions:Optional[Dict[str,str]] = None,
                 ):
        # for a thing it's keyword is it's name
        super().__init__(Keyword, Information, 
                         Editable = Editable,
                         Keyword_editable = Keyword_editable, 
                         Usage = Usage,
                         Model_name = Model_name,
                         Belongs_to = Belongs_to,
                        )
    def show(self):
        print(f'Keyword:{self.Keyword}\nInformation:{self.Information}\nEditable:{self.Editable}\nUsage:{self.Usage}')
    
    def get_information(self, viewer:Optional[Any] = None) -> str:
        return f'<{self.Keyword}>' + ': ' + self.Information
    
    def get_keyword(self) -> str:
        return f'<{self.Keyword}>'
    
    def destory(self):
        # pointers 1,Space objects.  2,Container
        # If the thing in the contiainer delete 
        # If 
        super().destory()

    def edit(self,
             edited_information:str): # approx $0.0001 each time if use gpt-3.5
        super().edit(edited_information) # self.information is already updated by Base class

    def systemic_parse(self, attemptation_action:'plan_system.Attemptation_Interactive_Action'):
        if attemptation_action.Success_fail_state:
            do_carry = self._get_decide_carry_func(attemptation_action)()['parsed_result']
            if do_carry:
                attemptation_action.Host_CHIBI.grab_item(self) # edit state information experience add and impression object delete
                # delete the impression objcet
            else:
                pass
        
    def interact_pipeline(self, attemptation_action:'plan_syste.Attemptation_Interactive_Action'):
        super().interact_pipeline(attemptation_action)

    def _get_decide_carry_func(self, attemptation_action:'plan_system.Attemptation_Interactive_Action'):
        @utils.Prompt_constructor_for_system(self.Model_name,
                                             Usage = self.Usage,
                                             parse_function_str = 'str_with_tuple')
        def _prompt_and_input():
            Prompt = f'''Based on the action description of {attemptation_action.Host_CHIBI.Name}, please answer the following questions. The specific action description is: {attemptation_action.detailed_action_string}.\nYou need to answer the following question based on the above action description. Guess logocally based on the action description, is the {self.get_keyword()} been taken away by {attemptation_action.Host_CHIBI.Name} after the action is finished, or the {self.get_keyword()} has been left at the region where it was?'''
            Input = f'''Please answer your question in a tuple with only one boolean element. If you think {self.get_keyword()} has been taken away by {attemptation_action.Host_CHIBI.Name} please answer (True), if the {self.get_keyword()} has been left where it was (not be taken away by {attemptation_action.Host_CHIBI.Name}) please answer (False).'''
            return Prompt, Input
        return _prompt_and_input
    
        
class Thing_container(CHIBI_Object):
    # TODO: cahnge all current containers into object
    def __init__(self,
                 Keyword:str,
                 Information:str,
                 All_objects:Dict[str,Thing],
                 Editable:bool = True,
                 Keyword_editable:bool = True,
                 Model_name:str = 'gpt-3.5-turbo-0125',
                 Usage:Optional['openai.openai_object.OpenAIObject'] = None,
                 Openable:bool = False, # directly
                 Belongs_to:Optional['Thing_container'] = None,
                ):
        for thing in All_objects.values():
            assert isinstance(thing, Thing), 'stored value in the input dict should be a type of Thing'
        self.Keyword = Keyword
        self.All_objects = All_objects 
        self.Model_name = Model_name
        if Usage is None:
            self.Usage = {'completion_tokens': 0,   # returned by parse function
                          'prompt_tokens':0,            # openai usage
                          'total_tokens': 0} 
        else:
            self.Usage = Usage
        for a_thing in All_objects.values():
            a_thing.Belongs_to = self
        self.Information = Information
        self.Editable = Editable
        self.Keyword_editable = Keyword_editable
        self.Openable = Openable
        self.Belongs_to = Belongs_to
        
    def object_retrieve(self, 
                       mode:str = 'Default',
                       prompt:Optional[str] = None,
                       Task:Optional[str] = None,
                       viewer:Optional[Any] = None) -> List[Thing]: 
        '''Only return items in this container'''
        if mode == 'Default':
            '''In default setting, the container will retrun all items inside and form a string'''
            return list(self.All_objects.values())
        elif mode == 'Return_string':
            return_str = f'{self.Keyword}:\n' + '\n'.join([i.get_information(viewer) for i in self.All_objects.values()])
            return return_str
        else:
            raise ValueError(f'{mode} not known for {type(self)}')
            
    def object_delete(self,
                      thing_to_be_deleted: Thing):
        keys_to_remove = []
        for single_thing in self.All_objects.values():
            if single_thing is thing_to_be_deleted:
                keys_to_remove.append(thing_to_be_deleted.Keyword)
                thing_to_be_deleted.Belongs_to = None
                break
                   
        for key in keys_to_remove:
            del self.All_objects[key]
            #print(f'{key} deleted from {self.Keyword}!')   
    
    def object_add(self, 
                   thing_to_be_added: Thing):
        if thing_to_be_added.Belongs_to is not None:
            thing_to_be_added.Belongs_to.object_delete(thing_to_be_added)
        self.All_objects.update({thing_to_be_added.Keyword:thing_to_be_added})
        thing_to_be_added.Belongs_to = self
    
    def show(self):
        print(f'Current container: {self.Keyword} have the following object: \n')
        for keyword,object in self.All_objects.items():
            print(f'---------------------- {keyword} -------------------')
            object.show()
  
    def edit(self,
             edited_information:str):
        if self.Editable:
            self.Information = edited_information
        else:
            assert self.Editable

    def get_information(self, viewer:Optional[Any] = None) -> str:
        '''Should provide all information in this object, return information structually in this object'''
        return self.Information

    def get_keyword(self):
        return f'<{self.Keyword}>'

    def destory(self):
        '''some thing happens when this thing is deleted'''
        # relase_all_container_stuff
        release_objects = list(self.All_objects.values())
        for single_container_stuff in release_objects:
            self.Belongs_to.object_add(single_container_stuff)
            single_container_stuff.Belongs_to = self.Belongs_to
        super().destory()

    def systemic_parse(self, attemptation_action:'plan_system.Attemptation_Interactive_Action'):
        if attemptation_action.Success_fail_state:
            do_open = self._get_decide_open_func(attemptation_action)()['parsed_result']
            if do_open:
                contained_stuffs = self.object_retrieve()
                if len(contained_stuffs) == 0:
                    open_experience = f'You opened the {self.Interactive_object.get_keyword()}, unfortunately the {self.get_keyword()} is empty, there nothing in it!'
                else:
                    keywords = [i.get_keyword() for i in contained_stuffs]
                    open_experience = f'''You opened the {self.get_keyword()}, you find the following items in the {self.get_keyword()}: {','.join(keywords)}.'''
                    for container_stuff in contained_stuffs:
                        # create impression for the object
                        assert isinstance(container_stuff, Thing), f'Currently, only Thing object can be placed into container, however {container_stuff.get_keyword()} is a {type(container_stuff)}'
                        impression_space = attemptation_action.Host_CHIBI.Space_manager.get_cur_space(space_type = 'impression')
                        
                        impression_object = Object_Impression(container_stuff.Keyword, 
                                                             f'You find a {container_stuff.Keyword} from {self.get_keyword()}, you haven\'t investigated it closely yet',
                                                             container_stuff,self,impression_space)
                attemptation_action.Host_CHIBI.Memory_stream.memory_add(open_experience)
            else:
                pass
        
    def interact_pipeline(self, attemptation_action:'plan_syste.Attemptation_Interactive_Action'):
        super().interact_pipeline(attemptation_action)

    def _get_decide_open_func(self, attemptation_action:'plan_system.Attemptation_Interactive_Action'):
        @utils.Prompt_constructor_for_system(self.Model_name,
                                             Usage = self.Usage,
                                             parse_function_str = 'str_with_tuple')
        def _prompt_and_input():
            Prompt = f'''Based on the action description of {attemptation_action.Host_CHIBI.Name}, please answer the following questions. The specific action description is: {attemptation_action.detailed_action_string}.\nYou need to analysis the above action description and answer the following quesion. After the action did {attemptation_action.Host_CHIBI.Name} managed to open the {self.get_keyword()} and the reason to do so is to get the items in the {self.get_keyword()}. Please following the steps to get your final answer.'''
            Input = f'''**Step1**: Did above action description mentioned verbs like live open, try opening, unlock...... that indicating {attemptation_action.Host_CHIBI.Name} try to get the item in the {self.get_keyword()}? Give your analysis.\n**Step2**Please answer your question in a tuple with only one boolean element, if {attemptation_action.Host_CHIBI.Name} intend to get the item in {self.get_keyword()} please answer (True), if {attemptation_action.Host_CHIBI.Name} did not intend to get the item in {self.get_keyword()} please answer (False).'''
            return Prompt, Input
        return _prompt_and_input

# -------------------------------Information_piece related classes------------------------------------
# -------------------------------Information_piece related classes------------------------------------
# -------------------------------Information_piece related classes------------------------------------
class Information_piece(CHIBI_Object):
    '''This is a pure information that can\'t be touch
       Currently only used in profiles'''
    def __init__(self,
                 Keyword:str,
                 Information:str,
                 Editable:bool = True,
                 Keyword_editable:bool = False,
                 Model_name:str = 'gpt-3.5-turbo-0125',
                 Usage:Optional['openai.openai_object.OpenAIObject'] = None,
                 Belongs_to:Optional["Information_piece_container"] = None,
                 ):
        super().__init__(Keyword, Information, 
                         Editable = Editable,
                         Keyword_editable = Keyword_editable, 
                         Usage = Usage,
                         Model_name = Model_name,
                         Belongs_to = Belongs_to)
        
    def edit(self,
             edited_information:str): # approx $0.0001 each time if use gpt-3.5
        super().edit(edited_information) # self.information is already updated by Base class
#         if self.Keyword_editable:
#             generated_summary = self._get_update_keyword_function(edited_information)()
#             if self.Usage is None:
#                 self.Usage = generated_summary['new_usage']
#             parsed_tuple = generated_summary['parsed_result']
#             if parsed_tuple[0]:# gpt thinks edit on this information will cause the keyword to change
#                 old_keyword = self.Keyword
#                 del self.Belongs_to.All_objects[old_keyword]
#                 self.Keyword = parsed_tuple[1]
#                 int_count = 1
#                 while self.Keyword in self.Belongs_to.All_objects:
#                     int_count += 1
#                     self.Keyword = f'{parsed_tuple[1]}{int_count}号'
#                 self.Belongs_to.All_objects.update({self.Keyword:self})
#             else:
#                 # gpt thinks origional keyword still align to edited information 
#                 pass
            
#     def _get_update_keyword_function(self,
#                                      edited_information:str) -> Callable[[],Callable[...,Any]]:
#         Prompt = f'''Some previously recorded information has been altered. First, you need to determine if the original index can still correspond to this altered information. If the original index cannot correspond to the new information, you need to generate a new index. Please output a tuple as the result, where the first element indicates whether the index needs to be modified and the second element is the modified new index, if necessary. Ensure that the subject of the index and the altered information match. Here are two examples:

# Example 1:
# <Input> Original index: My Address. Original information: My home is located at 13th Building, Donghua District, Room 212, surrounded by many subway and bus stations, which makes it very convenient to go home or to work. Altered description: My home is in 12th Building, Chaoyang District, and there are a lot of places to eat outside, which I find very satisfying.
# <Output>: (False, None)

# Example 2:
# <Input> Original index: Location of the Treasure. Original information: Senior students at the school told me that there is a treasure in the bushes of the back hill. Altered information: Senior students deceive the junior students by telling them there is a treasure in the back hill, and then they bully them when the junior students go there.
# <Output>: (True, "Senior Students' Treasure Trick")'''

#         Input = f'''The original index of this information: {self.Keyword}, the original information: {self.Information}, the changed information: {edited_information}'''
#         @utils.Prompt_constructor_for_system(self.Model_name,Usage = self.Usage)
#         def _update_keyword():
#             return Prompt, Input
#         return _update_keyword    
    
    def show(self):
        print(f'Keyword:{self.Keyword}\nInformation:{self.Information}\nEditable:{self.Editable}\nUsage:{self.Usage}')
    
    def get_information(self, viewer:Optional[Any] = None) -> str:
        return super().get_information(viewer)
    
    def get_keyword(self) -> str:
        return f'<{self.Keyword}>'
    
    def destory(self):
        pass

    def interact_pipeline(self):
        assert False, f'''currently {type(self)} is not interactable'''

class Information_piece_container(CHIBI_Object):
    '''Focus on profile like information eg:personalities, skills have, this is a hard coded and pre-organized profile'''
    def __init__(self,
                 Keyword:str,
                 All_objects:Dict[str,Information_piece],
                 Model_name:str = 'gpt-3.5-turbo-0125',
                 Usage:Optional[Dict[str,int]] = None,
                ):
        for information_piece in All_objects.values():
            assert isinstance(information_piece, Information_piece), 'stored value in the input dict should be a type of Thing'
        self.Keyword = Keyword
        self.All_objects = All_objects
        self.Model_name = Model_name
        if Usage is None:
            self.Usage = {'completion_tokens': 0,   # returned by parse function
                          'prompt_tokens':0,            # openai usage
                          'total_tokens': 0} 
        else:
            self.Usage = Usage
        for an_information_piece in All_objects.values():
            an_information_piece.Belongs_to = self
        
    def object_retrieve(self, 
                        mode:str = 'Default',
                        viewer:Optional[Any] = None,
                        prompt:Optional[str] = None,
                        Task:Optional[str] = None,) -> Union[List[Information_piece],str]: 
        '''if object in this contianer is more than certain value,
        to avoid super long prompt, 
        we should pick most significant objects to construct the prompt'''
        if mode == 'Default':
            '''In default setting, the container will retrun all All_objects inside and form a string'''
            return list(self.All_objects.values())
        elif mode == 'Return_string':
            return_str = f'{self.Keyword}:\n' + '\n'.join([i.get_information(viewer) for i in self.All_objects.values()])
            return return_str
        else:
            assert f'All_objects retrieve mode:{mode} is not supported'
    
    
    def object_delete(self,
                         information_piece_to_be_deleted: Information_piece):
        assert False, f'not implemented yet'
        pass

    def object_add(self, 
                      information_piece_to_be_added: Information_piece):
        self.All_objects.update({information_piece_to_be_added.Keyword:information_piece_to_be_added})
    
    def show(self):
        print(f'Current container: {self.Keyword} have the following object: \n')
        for keyword,object in self.All_objects.items():
            print(f'---------------------- {keyword} -------------------')
            object.show()
            
    def edit(self,
             edited_information:str):
        assert False, f'type(self) is not editable'

    def get_information(self,viewer:Optional[Any] = None) ->str:
        assert False, f'{type(self)} do not have information'

    def get_keyword(self):
        return f'<{self.Keyword}>'
        
    def destory(self):
        print(f'destory function for {type(self)} is not implemented')
        pass
        
    def interact_pipeline(self):
        assert False, f'''currently {type(self)} is not interactable'''

# ------------------------------- Space objects classes------------------------------------
# ------------------------------- Space objects classes------------------------------------
# ------------------------------- Space objects classes------------------------------------
class Edge_Base(CHIBI_Object, ABC):
    def __init__(self,
                 Information:Dict['Space_Base',str],
                 Editable:bool = False,
                 #Passable_dict: Optional[Dict['Space_Base',bool]] = None,
                 Usage:Optional[Dict[str,int]] = None,
                 Model_name:str = 'gpt-3.5-turbo-0125'):
        self.Information = Information
        self.Editable = Editable
        # if Passable_dict is None:
        #     Passable_dict = {}
        #     for key in self.Information.keys():
        #         Passable_dict.update({key:False})
        #     self.Passable_dict = Passable_dict
        # else:
        #     self.Passable_dict = Passable_dict
        assert len(Information) == 2, f'Information for {type(self)} should have exact 2 item with two vertices'
        self.Model_name = Model_name
        if Usage is None:
            self.Usage = {'completion_tokens': 0,   # returned by parse function
                          'prompt_tokens':0,            # openai usage
                          'total_tokens': 0} 
        else:
            self.Usage = Usage
        
    # @abstractmethod
    # def passable(self, view_space:Union[str, 'Space_Base']) -> Tuple[bool,str]:
    #     '''试图通过这条边移动的角色是否可以移动'''
    #     '''如果不能通过的话还需要返回原因'''
    #     return (True,)

    def get_two_end(self):
        return list(self.Information.keys())
            
    def edit(self, edit_information:str, editor:'CHIBI.CHIBI_Base'):
        # TODO 需要将这一部分。可能需要下下周再说了                           
        edit_space = editor.Space_manager.Cur_position
        self.Information[edit_space] = edit_information

class Edge_Double_Side(Edge_Base):
    def __init__(self,
                 Information:Dict['Space_Base',str],
                 Keyword:Optional[str] = None,
                 Obj_information:Optional[str] = None,
                 Keyword_editable:bool = True,
                 Model_name:str = 'gpt-3.5-turbo-0125',
                 Usage:'openai.openai_object.OpenAIObject' = None,
                 Belongs_to:Optional["Thing_container"] = None,
                 Editable:bool = True,
                 Passable_dict: Optional[Dict['Space_Base',bool]] = None
                 ):
        super().__init__(Information, Editable = Editable, Model_name = Model_name, Usage = Usage)
        self.Keyword = Keyword
        self.Obj_information = Obj_information
        self.Keyword_editable = Keyword_editable
        
        
        self.Belongs_to = Belongs_to # may not need this in the future
    # -------- Edge_Base interfaces
    # def passable(self,
    #              Pass_CHIBI:'CHIBI.CHIBI_Base') -> Tuple[bool,str]:
    #     '''This should be editable when edit are called'''
    #     CHIBI_cur_position_str = Pass_CHIBI.Space_manager.Cur_position
    #     CHIBI_cur_space = Pass_CHIBI.Space_Manager_System_Global.Vertices_dict[CHIBI_cur_position_str]
    #     return (self.Passable_dict[CHIBI_cur_space],self.get_information(CHIBI_cur_space))

    def edit(self,
             edited_information:str):
        '''edited_information is the modified new object information and now we need to align the object information to edge information for both side'''    
        two_end_space = self.get_two_end()
        for one_space in two_end_space:
            new_end_information = self._get_edit_edge_information_func(edited_information, one_space)()['parsed_result']
            self.Information[one_space] = new_end_information
        self.Obj_information = edited_information
    
    def _get_edit_edge_information_func(self, 
                                        new_description:str, 
                                        view_space):
        @utils.Prompt_constructor_for_system(self.Model_name,
                                             Usage = self.Usage,
                                             parse_function_str = None)
        def _prompt_input():
            Prompt = f'''You are a system component tasked with updating information about an object, {self.get_keyword()}, from two different perspectives. Currently, the information about this Edge object is as follows:\n{self.get_keyword()}: {self.Obj_information}.\nHowever, due to recent interactions, the state of this object has changed. The new state of the object is described as: {new_description}.\nOriginally, the description of this {self.get_keyword()} when viewed from the {view_space.get_keyword()} perspective was: {self.Information[view_space]}.'''

            Input = f'''With the given information, you are requested to update the description of the object {self.get_keyword()} after recent interactions, specifically from the perspective of {view_space.get_keyword()}. In your update, maintain the descriptive style of the original information. Include all unchanged details and integrate any modifications you believe have occurred as a result of the interaction. Your goal is to provide a comprehensive and updated view of the object from the specified perspective, reflecting both its original state and any recent changes.'''

    def show(self):
        '''different type of object can have different show method directly print out all information in this object'''
        print(f'this edge is an object it has the following information {self.Keyword}: {self.Obj_information}')
        iter_obj = iter(self.Information.keys())
        vertex_1 = next(iter_obj)
        vertex_2 = next(iter_obj)
        vertex1_str = vertex_1.Space_name
        vertex2_str = vertex_2.Space_name
        '''different type of object can have different show method directly print out all information in this object'''
        print(f'from {vertex1_str} to watch this edge the information is: {self.Information[vertex_1]}\nform {vertex2_str} watch have the following information {self.Information[vertex_2]}')
    
    def get_keyword(self):
        return f'<{self.Keyword}>'
    
    def destory(self):
        '''Edge object is currently not destoriable'''
        pass
        #self.Belongs_to.delete_delete(self)
        
    def get_information(self,viewer:Optional[Any])->str:
        assert viewer is not None, f'''for an edge double side this function should have a viewer'''
        if viewer != "system":
            # viewer should be CHIBI here
            view_space = viewer.Space_manager.get_cur_space()
            edge_information = self.Information[view_space]
            obj_information = self.Obj_information if self.Obj_information is not None else ''
            return edge_information+obj_information
        else:
            # viewer = 'system'
            return_str = ''
            return_str += f'''The information about the {self.get_keyword()}\n''' is self.Keyword is not None or ''
            two_end = self.get_two_end()
            return_str += f'''From {two_end[0].get_keyword()} to {two_end[1].get_keyword()} the edge information is: {self.Information[two_end[0]]}\n'''
            return_str += f'''From {two_end[1].get_keyword()} to {two_end[0].get_keyword()} the edge information is: {self.Information[two_end[1]]}\n'''
            return return_str

    def systemic_parse(self, attemptation_action:'plan_system.Attemptation_Interactive_Action'):
        if attemptation_action.Success_fail_state:
            cur_space_str = attemptation_action.Host_CHIBI.Space_manager.Cur_position
            try_pass = self._get_decide_pass_func(attemptation_action)()['parsed_result']
            if try_pass:
                two_end = self.get_two_end()
                move_to_space = two_end[0] if cur_space_str == two_end[1].Space_name else two_end[1]
                move_to_str = move_to_space.Space_name
                attemptation_action.Host_CHIBI.move(move_to_str)# already created an space_impression here
                effort_description = f'''You arrived at {move_to_space.get_keyword()}'''
                attemptation_action.Host_CHIBI.Memory_stream.memory_add(effort_description)
            else:
                pass
        
    def interact_pipeline(self, attemptation_action:'plan_syste.Attemptation_Interactive_Action'):
        super().interact_pipeline(attemptation_action)

    def _get_decide_pass_func(self, attemptation_action:'plan_system.Attemptation_Interactive_Action'):
        @utils.Prompt_constructor_for_system(self.Model_name,
                                             Usage = self.Usage,
                                             parse_function_str = 'str_with_tuple')
        def _prompt_and_input():
            Prompt = f'''Based on the action description of {attemptation_action.Host_CHIBI.Name}, please answer the following questions. The specific action description is: {attemptation_action.detailed_action_string}.\nYou need to analysis the above action description and answer the following quesion. After the action did {attemptation_action.Host_CHIBI.Name} stays at the same position(ie, the action described did not move {attemptation_action.Host_CHIBI.Name}'s position). Please following the steps to get your final answer.'''
            Input = f'''**Step1**: Did above action description mentioned verbs like live move, step ahead, pass through...... that indicating {attemptation_action.Host_CHIBI.Name} move from one position to another? Give your analysis.\n**Step2**Please answer your question in a tuple with only one boolean element, if {attemptation_action.Host_CHIBI.Name} moved to a new position please answer (True), if {attemptation_action.Host_CHIBI.Name} did not move to a new position please answer (False).'''
            return Prompt, Input
        return _prompt_and_input
            
class Space_Base(CHIBI_Object, ABC):
    def __init__(self, 
                 Space_name:str,
                 Space_connections:Optional[List[str]] = None,# first use string to store all rooms when they are initializing, Then we use dict to map name to Space object in manager object
                 All_objects:Optional[Dict[str,CHIBI_Object]] = None,
                 Editable:bool = False,
                 Visibility:int = 1,):
        '''A basic block for map structure, A space itself should be a container of Things, 
        and can also have other containers. Eg. A treasure box in a room, as well as single objects like desk and chair.'''
        if Space_connections is None:
            self.Space_connections = []
        else:
            self.Space_connections = Space_connections
        '''edges should be editable by agents'''
        self.Space_name = Space_name
        self.Keyword = Space_name # Space_name = Keyword
        self.Editable = Editable
        if All_objects is None:
            self.All_objects = {'Things':[],   
                                'Edges':[],
                                'CHIBIs':[],}
        else:
            for i in ['Things', 'Edges', 'CHIBIs']:
                assert i in All_objects, f'{i} not in given object dict. if no such object please input a empty list'
            self.All_objects = All_objects
        self.Visibility = Visibility
        
    @abstractmethod
    def object_delete(self, object_to_be_deleted:CHIBI_Object):
        # Since the key for impression object and key for real object should be the same so this function should also handle the delete for impression object
        for key in self.All_objects.keys():
            if object_to_be_deleted in self.All_objects[key]:
                self.All_objects[key].remove(object_to_be_deleted)
                #print(f'{object_to_be_deleted.get_keyword()} is removed from {self.get_keyword()}')
                return
        assert False, f'{object_to_be_deleted.get_keyword()} not in current Space: {self.get_keyword()}'

    @abstractmethod
    def object_add(self, object_to_be_add:CHIBI_Object):
        # currently all object added can only be CHIBI or thing
        # previous container delete the object and then current object add the item
        from CHIBI import CHIBI_Base
        import fixed_interactive_pipeline_objects as fixed_blocks
        if isinstance(object_to_be_add, CHIBI_Base):
            origional_position = object_to_be_add.Space_manager.Cur_position
            if origional_position != self.Space_name:
                object_to_be_add.Space_Manager_System_Global.Vertices_dict[origional_position].object_delete(object_to_be_add)
            self.All_objects['CHIBIs'].append(object_to_be_add)
            object_to_be_add.Space_manager.Cur_position = self.Space_name
            #print(f'{object_to_be_add.Name} moved from {origional_position} to {self.get_keyword()}')

        
        elif isinstance(object_to_be_add, Thing) or isinstance(object_to_be_add, Thing_container) or isinstance(object_to_be_add, fixed_blocks.Fixed_Interact_Pipeline_Object_Base) and not isinstance(object_to_be_add,fixed_blocks.Fixed_pipeline_Simple_Edge):
            if object_to_be_add.Belongs_to is not None:
                object_to_be_add.Belongs_to.object_delete(object_to_be_add)
            self.All_objects['Things'].append(object_to_be_add)
            object_to_be_add.Belongs_to = self
            #print(f'{object_to_be_add.get_keyword()} is now being placed to {self.get_keyword()}')
        elif isinstance(object_to_be_add, Edge_Double_Side) or isinstance(object_to_be_add, fixed_blocks.Fixed_pipeline_Simple_Edge):
            # if object_to_be_add.Belongs_to is not None:
            #     object_to_be_add.Belongs_to.object_delete(object_to_be_add)
            self.All_objects['Edges'].append(object_to_be_add)
        else:
            assert False, f'Currently object adding can only be CHIBI or Thing, but {object_to_be_add.get_keyword()} is a  {type(object_to_be_add)}'

class Space_System_global(Space_Base): #TODO: name should be changed into space system global later
    '''This is a space used for system space manager, this class stores ground truth and real data including all information'''
    def __init__(self,Space_name:str,
                 Space_connections:Optional[List[str]] = None,
                 All_objects:Optional[Dict[str,CHIBI_Object]] = None,
                 Editable:bool = False,
                 Visibility:int = 1,
                 Usage:Optional['openai.openai_object.OpenAIObject'] = None,
                 Model_name:str = 'gpt-3.5-turbo-0125',
                ):
        super().__init__(Space_name,Space_connections = Space_connections, 
                         All_objects = All_objects,
                         Editable = Editable, Visibility = Visibility)
        self.Model_name = Model_name
        self.Usage = Usage
        
    def edit(self):
        pass
    
    def update_space_keyword(self):
        pass
    
    def show(self):
        for object_type in self.All_objects:
            if len(self.All_objects[object_type]) == 0:
                continue
            print(f'----------cur space: {self.Space_name} for have following <{object_type}>-------------')
            for object in self.All_objects[object_type]:
                object.show()
                
    # Object interfaces
    def destory(self):
        pass

    def get_information(self, viewer:Optional[Any] = None):
        # todo update this
        return self.Space_name 
    
    def get_keyword(self):
        return f'<{self.Space_name}>'
    
    def retrieve_item_in_this_space(self,
                                    object_type:str = 'Things') -> List[Thing]:
        if object_type == 'All':
            return_objects = []
            for object_list in self.All_objects.values():
                return_objects.extend(object_list)
            return return_objects
        else:
            return self.All_objects[object_type]
        
    def object_add(self,
                   in_object:CHIBI_Object):
        super().object_add(in_object)
    
    def object_delete(self,
                      out_object:CHIBI_Object,):
        super().object_delete(out_object)
        
    def interact_pipeline(self):
        assert False, f'''currently {type(self)} is not interactable'''

# ------------------------------- Impression objects classes------------------------------------
# ------------------------------- Impression objects classes------------------------------------
# ------------------------------- Impression objects classes------------------------------------
    # currently the impression of a object should have the same keyword as real object
class Memory_piece(CHIBI_Object):
    def __init__(self,
                 Information: str,
                 Editable = False,
                 Belongs_to:Optional["Memory_stream"] = None,
                 Model_name:str = 'gpt-3.5-turbo-0125',
                 Embedding:Optional[list[float]] = None,
                 Time_passed:Union[datetime.timedelta,int] = datetime.timedelta(minutes = 0),
                 Importance_score:float = None, # range from 0~1
                 Memory_type:Optional[str] = None,
                 ):
        self.Information = Information
        self.Editable = False
        self.Belongs_to = Belongs_to
        self.Model_name = Model_name
        self.Embedding = Embedding
        if isinstance(Time_passed,int):
            Time_passed = datetime.timedelta(minutes = Time_passed)
        self.Time_passed = Time_passed # in minute
        self.Importance_score = Importance_score # can only be assigned by memory stream
        self.Memory_type = Memory_type

    def edit(self,
             edited_information:str):
        assert False, 'A single piece of memory can not be edited, they are happened facts'
        
    def show(self):
        print(self.Information)
    
    def get_information(self,viewer:Optional[Any] = None):
        return self.Information
    
    def get_keyword(self):
        assert False, 'A single piece of memory do not have a key word'
        
    def destory(self):
        pass

    def get_time_passed_in_hour(self)->int:
        total_seconds = self.Time_passed.total_seconds()
        return math.ceil(total_seconds/3600)
        
    def interact_pipeline(self):
        assert False, f'''currently {type(self)} is not interactable'''

class Assumption(CHIBI_Object):
    # Assumption is generated by abduction
    def __init__(self,
                 Information: str,
                 Belongs_to:Optional["Memory_stream_Base"] = None,
                 Model_name:str = 'gpt-3.5-turbo-0125',
                 Embedding:Optional[list[float]] = None,
                 Time_passed:Union[datetime.timedelta,int] = datetime.timedelta(minutes = 0),
                 Importance_score:float = None, # range from 0~10
                 ):
        self.Information = Information
        self.Belongs_to = Belongs_to
        self.Model_name = Model_name
        self.Embedding = Embedding
        if isinstance(Time_passed,int):
            Time_passed = datetime.timedelta(minutes = Time_passed)
        self.Time_passed = Time_passed # in minute
        self.Importance_score = Importance_score # can only be assigned by memory stream
        
    def edit(self,
             edited_information:str):
        self.Information = edited_information
        
    def show(self):
        print(self.Information)
    
    def get_information(self,viewer:Optional[Any] = None):
        return self.Information
    
    def get_keyword(self):
        assert False, 'A single Assumption do not have a key word'
        
    def destory(self):
        self.Belongs_to.memory_delete(self)
        self.Belongs_to = None
        

    def get_time_passed_in_hour(self)->int:
        total_seconds = self.Time_passed.total_seconds()
        return math.ceil(total_seconds/3600)
        
    def interact_pipeline(self):
        assert False, f'''currently {type(self)} is not interactable'''

class Plan_piece(CHIBI_Object):
    # based on assumption, CHIBI generate a plan to guide aciton selection, explore new observation to generate new rule or use current rule to exploit.
    # TODO: 1, currently use this in the memorystream, later in the future should update plan node in plan system to have more detailed loop for complex task
    def __init__(self,
                 Information: str,
                 Belongs_to:Optional["Memory_stream_Base"] = None,
                 Model_name:str = 'gpt-3.5-turbo-0125',
                 Embedding:Optional[list[float]] = None,
                 Time_passed:Union[datetime.timedelta,int] = datetime.timedelta(minutes = 0),
                 Importance_score:float = None, # range from 0~10
                 ):
        self.Information = Information
        self.Belongs_to = Belongs_to
        self.Model_name = Model_name
        self.Embedding = Embedding
        if isinstance(Time_passed,int):
            Time_passed = datetime.timedelta(minutes = Time_passed)
        self.Time_passed = Time_passed # in minute
        self.Importance_score = Importance_score # can only be assigned by memory stream
        
    def edit(self,
             edited_information:str):
        self.Information = edited_information
        
    def show(self):
        print(self.Information)
    
    def get_information(self,viewer:Optional[Any] = None):
        return self.Information
    
    def get_keyword(self):
        assert False, 'A single Plan_piece do not have a key word'
        
    def destory(self):
        self.Belongs_to.memory_delete(self)
        self.Belongs_to = None
        

    def get_time_passed_in_hour(self)->int:
        total_seconds = self.Time_passed.total_seconds()
        return math.ceil(total_seconds/3600)
        
    def interact_pipeline(self):
        assert False, f'''currently {type(self)} is not interactable'''

class Object_Impression(CHIBI_Object):
    '''This is a class that keep track of an item, used for Space_memory, 
    CHIBI should able to create a impression of this object, and when CHIBI is thinking about this Object when planning
    he should retrieve Thing_memory rather than Thing directly'''
    def __init__(self,
                 Keyword:str,
                 Information:str, # if this is an edge ,just describe with information, nothing else needed no double side information should be considered
                 Impression_of:CHIBI_Object,
                 Belongs_to:Union['Space_CHIBI_impression'],
                 Impression_space:'Space_CHIBI_impression',
                 Editable:bool = True,
                 Keyword_editable:bool = True,
                 Model_name:str = 'gpt-3.5-turbo-0125',
                 Usage:Optional[Dict[str, int]] = None,
                 Embedding:Optional[List[float]] = None,
                 Need_embedding:bool = True,
                 ):
        self.Keyword = Keyword
        self.Information = Information
        self.Impression_of = Impression_of
        assert isinstance(Impression_of, CHIBI_Object), f'This Thing_memory should be an impression of a CHIBI_Object object rather than {type(Impression_of)}'
        self.Editable = Editable
        self.Keyword_editable = Keyword_editable
        self.Model_name = Model_name
        if Usage is None:
            self.Usage = {'completion_tokens': 0,   
                          'prompt_tokens':0,
                          'total_tokens': 0} 
        else:
            self.Usage = Usage
        self.Belongs_to = Belongs_to
        self.Impression_space = Impression_space

        self.Need_embedding = Need_embedding
        
        if self.Need_embedding:
            if Embedding is None:
                object_string = f'''{self.Keyword}: {self.Information}'''
                self.Embedding = utils.get_embedding(object_string)[0]
            else:
                self.Embedding = Embedding
        
    def edit(self,
             edited_information:str):
        self.Information = edited_information
        object_string = f'''{self.Keyword}: {self.Information}'''
        if self.Need_embedding:
            self.Embedding = utils.get_embedding(object_string)[0]
            
    def show(self):
        '''different type of object can have different show method'''
        pass

    def get_information(self, viewer:Optional[Any] = None) -> str:
        return f'<{self.Keyword}>' + ': ' + self.Information

    def get_keyword(self):
        return f'<{self.Keyword}>'

    def destory(self):
        '''some thing happens when this thing is deleted'''
        if isinstance(self.Belongs_to, Space_CHIBI_impression):
            self.Belongs_to.object_delete(self)

    def interact_pipeline(self):
        assert False, f'''currently {type(self)} is not interactable'''
        
class Space_CHIBI_impression(Space_Base):
    '''This is a space used for CHIBI space manager, this class stores information gathered from discorver of CHIBI, it may not be true and correct, but they are the outcome of the CHIBI's perception'''
    def __init__(self,Space_name:str,
                 Impression_of:Space_System_global,
                 Space_connections:List[str] = None,
                 All_objects:Dict[str,List[Object_Impression]] = None, 
                 Editable:bool = True,
                 Visibility:int = 1,
                 Usage:Optional['openai.openai_object.OpenAIObject'] = None,
                 Model_name:str = 'gpt-3.5-turbo-0125',
                 Overview_description:str = None
                 ):
        # TODO: 再考虑一下每个CHIBI的Space manager中是否有必要有Memory_stream_Space 这个类 还是需要的 但是不需要记录那么详细的信息，只需要记住哪个房间有哪些object就可以了
        self.Space_name = Space_name
        self.Space_connections = Space_connections
        if All_objects is None:
            self.All_objects = []
        else:
            self.All_objects = All_objects # should contain Thing_impression_object rather than real Thing
        self.Editable = Editable
        self.Visibility = Visibility 
        self.Model_name = Model_name
        self.Usage = Usage
        self.Impression_of = Impression_of
        self.Overview_description = Overview_description
        
    def edit(self):
        pass
    
    def update_space_keyword(self):
        pass
    
    def show(self):
        pass 
        
    # Object interfaces
    def destory(self):
        pass
    
    def get_information(self, viewer:Optional[Any] = None):
        assert False, f'{type(self)} get_information is not implement'
    
    def get_keyword(self):
        return f'<{self.Space_name}>'
        
    def object_add(self,
                   in_object:CHIBI_Object):
        assert isinstance(in_object, Object_Impression), f'''{type(self)} can only add impression object into it'''
        self.All_objects.append(in_object)
    
    def object_delete(self,
                      out_object:CHIBI_Object):
        assert isinstance(out_object, Object_Impression), f'''{type(self)} can only delete impression object into it'''
        self.All_objects.remove(out_object)

    def interact_pipeline(self):
        assert False, f'''currently {type(self)} is not interactable'''
# --------------------------------------------helper class---------------------------------------------
class Block_helper:
    Model_name = 'gpt-3.5-turbo-0125'
    @staticmethod
    def create_Thing_with_colon(input_str:str,
                                Editable:bool = True,
                                Keyword_editable:bool = True,
                                Model_name:str = None) -> Thing:
        if Model_name is None:
            Model_name = Block_helper.Model_name
        # create Thing with legacy file
        assert type(input_str) == str, "input should be a single string"
        splited_text = re.split('[:：]', input_str)

        return Thing(splited_text[0],splited_text[1],Editable = Editable, Keyword_editable = Keyword_editable, Model_name = Model_name)
    
    @staticmethod
    def create_Information_piece_with_tuple(input_tuple:tuple,
                                            Editable:bool = True,
                                            Keyword_editable:bool = False,
                                            Model_name:str = None) -> Information_piece:
        if Model_name is None:
            Model_name = Block_helper.Model_name
        # Create Information piece with legacy file
        assert len(input_tuple) == 2, "input_tuple can only have one pair at a time"
        assert type(input_tuple[0]) == str, "keyword should be a str."
        keyword = input_tuple[0]
        return Information_piece(keyword, input_tuple[1], Editable = Editable, Keyword_editable = Keyword_editable, Model_name = Model_name)

    