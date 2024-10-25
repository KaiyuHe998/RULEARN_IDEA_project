# basics
from typing import *
from abc import ABC, abstractmethod
from collections import deque
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

# CHIBI framework components
import world_basic_blocks as blocks
import fixed_interactive_pipeline_objects as fixed_blocks
import utils
import memory_stream
# ----------------------------------------------------Space class ----------------------------------------------------
# ----------------------------------------------------Space class ----------------------------------------------------
# ----------------------------------------------------Space class ----------------------------------------------------



# ---------------------------------------------Manager class ----------------------------------------------------
class Space_Manager_Base(ABC):
    '''System level Space manager'''
    def __init__(self, 
                 Vertices_dict:Dict[str,blocks.Space_System_global],
                 #Edges:Optional[Dict[str,Dict[str,Union[str,Edge_Simple]]]] = None, # should be transformed into Edge class in init     ):
                ):
        self.label_colors = {
                        'Edge': 'cyan',
                        'Vertex': 'gold'
                        }
        self.update_space_map(Vertices_dict = Vertices_dict)
        
    def update_space_map(self, 
                         Vertices_dict:Optional[Dict[str,blocks.Space_System_global]] = None):
        '''do some update like add vertice and edge to Vertices Adjancency list'''
        '''When passing edges to the update method make sure to copy existing edges unless you want to delete edges'''
        # Currently edges are saved in space object
        # In first edition, by default we let all edges be bidirectional
        # This function only update Graph structure
        if Vertices_dict is None:
            pass
        else:
            self.Vertices_dict = Vertices_dict

        Edges = {}
        for vertex in self.Vertices_dict.values():
            edges = []
            edges.extend(vertex.All_objects['Edges'])
            for edge in edges:
                if isinstance(edge, fixed_blocks.Fixed_pipeline_Simple_Edge):
                    set_key = frozenset(edge.Connected_two_space)
                    Edges.update({set_key:edge})
                else:
                    iter_obj = iter(edge.Information.keys())
                    end_1 = next(iter_obj)
                    end_2 = next(iter_obj)
                    set_key = frozenset([end_1,end_2])
                    Edges.update({set_key:edge})
        self.Edges = Edges

        # TODO： check for connected room there at least have an edge between them
        self.index_dict = []
        self.Adjacency_list = {}
        for tem_space in self.Vertices_dict.values():
            self.Adjacency_list.update({tem_space.Space_name:tem_space.Space_connections}) # strings
        G = nx.DiGraph()

        for item in self.Edges.items():
            edge_space_iter = iter(item[0])
            end_1 = next(edge_space_iter)
            end_2 = next(edge_space_iter)
            G.add_edge(end_1.Space_name, end_2.Space_name, object = item[1], label = 'Edge')
            G.add_edge(end_2.Space_name, end_1.Space_name, object = item[1], label = 'Edge')
        self.G = G

    @abstractmethod
    def find_path(self, 
                  start_vertex: str, 
                  end_vertex: str,
                  plot_path:bool = False,
                  mode:str = 'bfs') -> List['str']:
        if mode == 'bfs':
            '''default method to get the path is bfs search'''
            assert start_vertex in self.Vertices_dict.keys(), 'given start_vertex name is not in the current map'
            assert end_vertex in self.Vertices_dict.keys(), 'given end_vertex name is not in the current map'

            visited = set()  # keep track of vertices visited
            queue = deque([start_vertex])  # init a queue
            visited.add(start_vertex)  # start_vertex is visited

            # keep track of predecessors of current vertex
            predecessors = {start_vertex: None}

            while queue:
                cur_vertex = queue.popleft()

                if cur_vertex == end_vertex:
                    break  # find the end point break search

                # get all neighbors
                if cur_vertex in self.Vertices_dict:
                    for neighbour in self.Vertices_dict[cur_vertex].Space_connections:
                        if neighbour not in visited:
                            visited.add(neighbour)
                            queue.append(neighbour)
                            predecessors[neighbour] = cur_vertex  # set predecessor
            # print path
            if end_vertex not in predecessors:
                print(f'Given start and end vertex are not connected in the graph')
                return []
            else:
                path = []
                current = end_vertex
                while current is not None:
                    path.insert(0, current)  
                    current = predecessors[current]
                if plot_path:
                    path_edges = []
                    for i in range(1,len(path)):
                        path_edges.append((path[i-1],path[i]))
                        path_edges.append((path[i],path[i-1]))
                    edge_colors = ["red" if e in path_edges else self.label_colors['Edge'] for e in self.G.edges()]
                    node_colors = ["orange" if n in path else self.label_colors['Vertex'] for n in self.G.nodes()]
                    self.draw_graph(node_color = node_colors, edge_color = edge_colors)
                return path
                
    def draw_graph(self,
                   node_size:float = 500, 
                   font_size:float = 8,
                   figure_size:tuple = (10,8),
                   node_color:Optional[str] = None,
                   edge_color:Optional[str] = None 
                  ):
        pos = nx.kamada_kawai_layout(self.G)
        if node_color is None:
            node_color = self.label_colors['Vertex']
        if edge_color is None:
            edge_color = self.label_colors['Edge']
        plt.figure(figsize=figure_size)
        
        nx.draw(self.G, with_labels=True, node_color=node_color, 
                edge_color=edge_color, node_size=node_size, font_size=font_size) 
        plt.show()

    def get_space_according_Visibility(self):
        # currently not in use
        pass

class Space_Manager_System(Space_Manager_Base):
    '''System level Space manager'''
    def __init__(self, 
                 Vertices_dict:Dict[str,blocks.Space_System_global],
                 Edges:Optional[Dict[str,Dict[str,str]]] = None,
                 ):
        super().__init__(Vertices_dict)
        
        
    def draw_graph(self,
                   node_size:float = 500, 
                   font_size:float = 8,
                   figure_size:tuple = (10,8),
                   node_color:str = 'lightblue',
                   edge_color:str = 'gray'):
        super().draw_graph(node_size = node_size, font_size = font_size, figure_size = figure_size, 
                           node_color = node_color, edge_color = edge_color)
    def find_path(self, 
                  start_vertex: str, 
                  end_vertex: str,
                  plot_path:bool = False,
                  mode:str = 'bfs') -> List['str']:
        path = super().find_path(start_vertex,end_vertex,plot_path = plot_path, mode = mode)
        return path
        
    def retrivev_all_CHIBI_Objects_in_this_space(self,space_name:str) -> Dict[str,Any]:
        #TODO 在CHIBI的object里面也加上这个因为这样的话这个CHIBI就可以在
        all_CHIBI_obj_list = []
        focal_space = self.Vertices_dict[space_name]
        all_Thing_list = focal_space.retrieve_item_in_this_space()
        all_CHIBIs_list = list(focal_space.CHIBIs.values()) # 暂时不考虑隔着场景影响到相邻场景的CHIBI的情况
        all_Spaces = [focal_space]
        all_Spaces.extend([self.Vertices_dict[space_name_str] for space_name_str in focal_space.Space_connections])
        for chibi in all_CHIBIs_list:
            all_CHIBI_obj_list.extend(chibi.Profile.get_all_items()) #CHIBI backpack items object
        all_CHIBI_obj_list.extend(all_Thing_list)                    #CHIBI all things in the Space
        all_CHIBI_obj_list.extend(all_Spaces)                        #CHIBI all Space objects
        all_CHIBI_obj_list.extend(all_CHIBIs_list)                   #all CHIBI objects
        return {obj.get_keyword():obj for obj in all_CHIBI_obj_list}

    def get_all_Space_Thing_objects(self,
                                   space_name:str):
        return self.Vertices_dict[space_name].retrieve_item_in_this_space()
        
        
        
class Space_Manager_CHIBI(Space_Manager_Base):
    '''Agent level Space manager, handling the impressions(Space, and other object)'''
    def __init__(self,
                 Host_CHIBI:"CHIBI.CHIBI_Base",
                 Init_position:str,
                 Edges:Optional[Dict[str,Dict[str,Union[blocks.Edge_Base,str]]]] = None,
                 Vertices_dict:Optional[Dict[str,blocks.Space_System_global]] = None,
                 ):
        self.Cur_position = Init_position
        self.Host_CHIBI = Host_CHIBI
        self.Name = Host_CHIBI.Name
        if Vertices_dict is None:
            self.Vertices_dict = {}
        else:
            self.Vertices_dict = Vertices_dict
        if len(Vertices_dict) == 0:
            self.create_new_space_CHIBI_impression(Init_position)
        super().__init__(self.Vertices_dict)

    # 对于记忆中的地图信息。现在暂时没有edge impression的类
    # TODO 这里可能需要加上Edge的impression类
    def update_space_map(self,
                         Vertices_dict:Optional[Dict[str, blocks.Space_CHIBI_impression]],
                         Edges:Optional[Dict[str,Dict[str,Union[blocks.Edge_Base,str]]]] = None,):
        if Vertices_dict is None:
            pass
        else:
            self.Vertices_dict = Vertices_dict
        self.index_dict = []
        self.Adjacency_list = {}
        for tem_space in self.Vertices_dict.values():
            self.Adjacency_list.update({tem_space.Space_name:tem_space.Space_connections})
        G = nx.Graph()
        for vertex_name in self.Vertices_dict.keys():
            G.add_node(vertex_name)
            for connected_space_name in self.Vertices_dict[vertex_name].Space_connections:
                G.add_edge(vertex_name, connected_space_name)
        self.G = G
        
    def get_cur_space(self, 
                      space_type:str = 'real'):
        if space_type == 'real':
            return self.Host_CHIBI.Space_Manager_System_Global.Vertices_dict[self.Cur_position]
        elif space_type == 'impression':
            return self.Vertices_dict[self.Cur_position]
        else:
            assert False, f'''The input space_type should be real or impression'''
        
    def get_all_real_Edges(self):
        cur_space_str = self.Host_CHIBI.Space_manager.Cur_position
        cur_space_real = self.Host_CHIBI.Space_Manager_System_Global.Vertices_dict[cur_space_str]
        connected_spaces = [self.Host_CHIBI.Space_Manager_System_Global.Vertices_dict[space_str] for space_str in cur_space_real.Space_connections]
        all_edges = []
        for connected_space_real in connected_spaces:
            key = frozenset([cur_space_real,connected_space_real])
            edge = self.Host_CHIBI.Space_Manager_System_Global.Edges[key]
            all_edges.append(edge)
        return all_edges

    def find_path(self, 
                  start_vertex: str, 
                  end_vertex: str,
                  plot_path:bool = False,
                  mode:str = 'bfs') -> List['str']:
        assert end_vertex in self.Vertices_dict, f'{self.Host_CHIBI.Name}从来都没到达过{end_vertex},并不知道如何前往'
        return_path = super().find_path(start_vertex,end_vertex,plot_path = plot_path, mode = mode)
        return return_path

    def create_new_space_CHIBI_impression(self,
                                      create_space_name:str):
        #TODO 加上CHIBI自己对于这个空间的想法和实验
        assert create_space_name in self.Host_CHIBI.Space_Manager_System_Global.Vertices_dict, f'试图创建的Space {create_space_name} 并不存在'
        cur_space_system = self.Host_CHIBI.Space_Manager_System_Global.Vertices_dict[create_space_name]
        new_space_memory = blocks.Space_CHIBI_impression(cur_space_system.Space_name,
                                                  cur_space_system,
                                                  Space_connections = cur_space_system.Space_connections,)
        self.Vertices_dict.update({new_space_memory.Space_name:new_space_memory})
        self.update_space_map(self.Vertices_dict,)
        #new_space_CHIBI_impression = Space_CHIBI_impression()
    
    def create_impression_of_an_object(self, 
                                     impression_of: blocks.CHIBI_Object,
                                     impression_space: blocks.Space_CHIBI_impression,
                                     mode = 'Default',
                                     belongs_to: Optional[Union[blocks.Space_CHIBI_impression, blocks.Object_Impression]] = None,)->blocks.Object_Impression:
                                     # if the object is not in a container or something belongs_to == impression_space) -> blocks.Object_Impression:
        #TODO: based on agent profile and information create impression description
        '''Impression should have the same keyword as the origional object'''
        
        if mode == 'Default':
            # in Default mood the generated impression have the default description
            default_information = f'It looks like an ordinary {impression_of.Keyword}, there is no more detailed information about the item.'
            if belongs_to is None: # can be a container impression and the information should be 
                belongs_to = impression_space
            else:
                default_information = f'It looks like an ordinary {impression_of.Keyword}, the {impression_of.Keyword} is placed in {belongs_to.get_keyword()}, there is no more detailed information about the item.'
            need_embedding = self.Host_CHIBI.CHIBI_type == 'GPT_agent'
            new_thing_memory = blocks.Object_Impression(impression_of.Keyword, 
                                                        default_information, 
                                                        impression_of,
                                                        belongs_to, 
                                                        impression_space,
                                                        Need_embedding = need_embedding)
            impression_space.object_add(new_thing_memory)
            #return new_thing_memory
        else:
            assert False, f'mode = {mode} is not supported now'

    def update_impression_in_cur_space(self):
        '''Need to be called when CHIBI get into a new space'''
        cur_space_impression = self.get_cur_space()
        cur_space_real = self.Host_CHIBI.Space_Manager_System_Global.Vertices_dict[cur_space.Space_name]
        # 真实物体应该是impression物体的子集， 如果说有些impression没有对应的实际物体 那么就应该是被挪走了，这种逻辑之后再处理，一般来说只有你需要这个物体的时候你才会察觉到这个物体不在了
        # We need to make sure that all real object in current space have it's impression
        # TODO: we need to use ebmedding to select top object to have impression
        all_impressions = []
        all_real_objects = []
        for object_type_list_impression in cur_space_impression.All_objects.values():
            all_impressions.extend(object_type_list)
        for object_type_list_real in cur_space_real.All_objects.values():
            all_real_objects.extend(object_type_list_real)
        impression_covered_real_objects = [i.impression_of for i in all_impressions]
        real_object_no_impression = set(all_real_objects) - set(impression_covered_real_objects) 
        if len(real_object_no_impression) == 0:
            return
        else:
            for real_object in real_object_no_impression:
                self.create_impression_of_an_object(real_object, cur_space_impression) 

    def find_impression_object(self, real_object:blocks.CHIBI_Object):
        # TODO any other faster way?
        for space in self.Vertices_dict.values():
            for single_object_impression in space.All_objects:
                if single_object_impression.Impression_of is real_object:
                    return single_object_impression


    def update_space_description(self, generate_method:str = 'gpt'):
        '''Can only update information of a space that CHIBI currently in.'''
        cur_impression_space = self.get_cur_space(space_type = 'impression')
        # 之后应该还要改成类似于State一样累积改变的，但是现在就暂时维持这样吧
        
        if cur_impression_space.Overview_description is None:
            if generate_method == 'gpt':
                # generated_str = self._get_generate_space_description_func(cur_impression_space)()['parsed_result']
                # cur_impression_space.Overview_description = generated_str
                # experience_string = f'You looked at the {self.Cur_position}, and here is the overview of this space:{generated_str}'
                # Currently not in use
                assert False, f'''Current gpt generated space description is not in use'''
            elif generate_method == 'system': # for human user
                experience_string = f'''You entered {cur_impression_space.Space_name}. Currently there are following items in this room: {','.join(i.get_keyword() for i in cur_impression_space.All_objects)}'''
                cur_impression_space.Overview_description = experience_string
            else:
                assert False, f'''{generate_method} is not a supported method.'''
            self.Host_CHIBI.Memory_stream.memory_add(experience_string, Memory_type = 'Observation', Importance_score = 10)
        # TODO: If object in impression space is not synchronised with real space, then need to add a new description that mention the difference
            
            
    # def _get_generate_space_description_func(self, impression_space:blocks.Space_CHIBI_impression):
    #     space_string = f'''Room Name:{impression_space.Space_name}\nItems:{','.join(i.get_keyword() for i in impression_space.All_objects)}'''
    #     @utils.Prompt_constructor_for_system(self.Host_CHIBI.Model_name,
    #                                          parse_function_str = None,
    #                                          Usage = self.Host_CHIBI.Usage)
    #     def _input_and_prompt():
    #         Prompt = '''Describe the contents of the room using only the items listed. Do not add or invent any details about items not described. Use simple language to summarize the room's contents:'''
    #         Input = f'''Below is the description of the space that you need to describe.\n{space_string}\n\n Following is a example answer that you can referred to:\n**User Input** Room Name: Living Room\n**Items**: <Large Beige sofa>,<Red Wooden Armchair>,<Green Wooden Armchair>,<Glass Coffee Table>,<Small Wool Rug>,<Wall-mounted Television>,<Bookshelf Filled With Books>\n\n**Your answer**:In the living room, there is a large beige sofa, two red armchairs, a glass coffee table, and a small wool rug on the floor. A television is mounted on the wall, and there is a bookshelf filled with various books.'''
    #         return Prompt, Input
    #     return _input_and_prompt
        
# Space helper class
class Space_helper:
    @staticmethod
    def generate_all_rooms_new_file(Adjacency_list:Dict[str,List[str]],
                                    Double_Side_Edge_Objects:Optional[Dict[str,Dict[str,str]]],
                                    Space_items:Optional[Dict[str,List[str]]] = None,
                                    Thing_containers:Optional[Dict[str,List[Dict[str,List[str]]]]] = None,
                                    Fixed_pipeline_map_items:Optional[Dict[str, Any]] = None,
                                    Model_name:str = 'gpt-3.5-turbo-0125'):
        Spaces = {} # return spaces
        Space_names = []
        for i in Adjacency_list:
            if i not in Space_names:
                Space_names.append(i)
            for j in Adjacency_list[i]:
                if j not in Space_names:
                    Space_names.append(j)
        if Space_items is None:
            Space_items = {}
        if Thing_containers is None:
            Thing_containers = {}

        for Space_name in Space_names:
            if Space_name in Adjacency_list:
                # outdegree rooms from adjacency list
                connected_rooms = Adjacency_list[Space_name]
            else:
                connected_rooms = []
    
            # indegree, for spaces that can only enter without exit, we can use Edge_Object
            for tem_space_name, tem_space in Adjacency_list.items():
                if Space_name in Adjacency_list[tem_space_name]:
                    connected_rooms.append(tem_space_name)
    
            # create things
            if Space_name in Space_items:
                if type(Space_items[Space_name]) == list:
                    single_things = [blocks.Block_helper.create_Thing_with_colon(i,Model_name = Model_name) for i in Space_items[Space_name]]
            else:
                single_things = []
            
            # create fixed things
            if Space_name in Fixed_pipeline_map_items:
                fixed_objects = Fixed_pipeline_map_items[Space_name]
                for single_fixed_object_dict in fixed_objects:
                    single_fixed_object = fixed_blocks.Fixed_Block_helper.create_fixed_object_with_dict(single_fixed_object_dict)
                    single_things.append(single_fixed_object)
            
            # create containers
            space_containers = []
            if Space_name in Thing_containers:
                for item in Thing_containers[Space_name].items():
                    container_obj_information = item[0]
                    contained_obj_list = item[1]
                    container_keyword, container_information = container_obj_information.split(':')
                    things_in_container = [blocks.Block_helper.create_Thing_with_colon(i,Model_name = Model_name) for i in contained_obj_list]
                    things_in_container = {a_thing.Keyword:a_thing for a_thing in things_in_container}
                    cur_container = blocks.Thing_container(container_keyword, container_information, things_in_container)
                    space_containers.append(cur_container)
        
            all_things = []
            all_things.extend(space_containers)
            all_things.extend(single_things)
            All_objects = {'Things':[],   
                           'Edges':[],
                           'CHIBIs':[],}

            cur_space = blocks.Space_System_global(Space_name,Space_connections = connected_rooms, 
                                            All_objects = All_objects,
                                            Model_name = Model_name)   
            for single_item in all_things:
                cur_space.object_add(single_item)

            Spaces.update({cur_space.Space_name: cur_space})
        
        # adding edges
        for space1_str in Double_Side_Edge_Objects:
            space1 = Spaces[space1_str]
            for space2_str in Double_Side_Edge_Objects[space1_str]:
                space2 = Spaces[space2_str]
                Obj_keyword, Obj_information, space1_information, space2_information = Double_Side_Edge_Objects[space1_str][space2_str].split(':')
                Information = {space1:space1_information,
                               space2:space2_information}
                new_edge_object = blocks.Edge_Double_Side(Information, Obj_keyword, Obj_information)
                space1.object_add(new_edge_object)
                space2.object_add(new_edge_object)
        return Spaces

    @staticmethod
    def generate_all_room_with_database(Map:Dict[str,List[str]],
                                        Space_items:Dict[str,List[int]],
                                        Space_item_containers:Dict[str,Dict[int,int]],
                                        Edges:Dict[str,Dict[str,List[int]]],
                                        Model_name:Optional[str] = None,
                                        ):
        Spaces = {} # return spaces
        Space_names = []
        for i in Map:
            if i not in Space_names:
                Space_names.append(i)
            for j in Map[i]:
                if j not in Space_names:
                    Space_names.append(j)
                    
        for Space_name in Space_names:
            if Space_name in Map:
                # outdegree rooms from adjacency list
                connected_rooms = Map[Space_name]
            else:
                connected_rooms = []
                
            for tem_space_name, tem_space in Map.items():
                if Space_name in Map[tem_space_name]:
                    connected_rooms.append(tem_space_name) #in degree spaces (Since currently all edge objects are double sided)
            # now connected_rooms contained all space that connected to Space_name
            # Step1: Create normal things'
            all_things_in_cur_space = []
            if Space_name in Space_items:
                for single_object_id in Space_items[Space_name]:
                    new_object = fixed_blocks.Fixed_Block_helper.create_fixed_object_with_database(single_object_id, Model_name = Model_name)
                    all_things_in_cur_space.append(new_object)

            # Step2: Create container
            if Space_name in Space_item_containers:
                for container_item in Space_item_containers[Space_name].items():
                    container_object_id = container_item[0]
                    contained_object_id_list = container_item[1]
                    new_container_object = fixed_blocks.Fixed_Block_helper.create_fixed_container_with_database(container_object_id, contained_object_id_list, Model_name = Model_name)
                    all_things_in_cur_space.append(new_container_object)

            All_objects = {'Things':[],   
                           'Edges':[],
                           'CHIBIs':[],}

            # Step3: Create Space and add things into it
            cur_space = blocks.Space_System_global(Space_name,Space_connections = connected_rooms, 
                                            All_objects = All_objects,
                                            Model_name = Model_name)   
            for single_item in all_things_in_cur_space:
                #print(f'current adding {single_item.get_keyword()} into {Space_name}')
                cur_space.object_add(single_item)
            Spaces.update({cur_space.Space_name: cur_space})

        # Step4: Create edge and add them into related two spaces
        for Space1_name in Edges:# Currently all edge in Edges are int id for fixed pipeline object 
            Space1 = Spaces[Space1_name]
            for Space2_name in Edges[Space1_name]:
                Space2 = Spaces[Space2_name]
                for edge_object_id in Edges[Space1_name][Space2_name]:
                    connected_edge_list = [Space1, Space2]
                    new_edge_object = fixed_blocks.Fixed_Block_helper.create_fixed_edge_with_database(edge_object_id, connected_edge_list, Model_name = Model_name)
                    Space1.object_add(new_edge_object)
                    Space2.object_add(new_edge_object)
                    
        return Spaces

        