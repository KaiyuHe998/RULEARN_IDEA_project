import copy


# --------------------------------------Function operator Puzzle--------------------------------------
# --------------------------------------Function operator Puzzle--------------------------------------
# --------------------------------------Function operator Puzzle--------------------------------------
Function_operator_puzzles = {
    # Level1 puzzles, clear assumption can be found in the scene，Only deduction is evaluated in this step
    'Level1':{
        'puzzle1':{'Map':{'Puzzle room':[]},
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new information and keyword
                   'Space_items': {'Puzzle room':[{'Object_id':68,'Special_label':'''True,6'''},69]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'''You are {Name}, you need to assign values to the functions displayed on the <Computer>, determin the value of 'a' and input it into the <Code secured door> to open it.''',
                             'Items':{},
                             'Init_position':'Puzzle room',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50
                   },
        'puzzle2':{'Map':{'Puzzle room':[]},
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new information and keyword
                   'Space_items': {'Puzzle room':[{'Object_id':68,'Special_label':'''True,1*x'''},69]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'''You are {Name}, you need to assign values to the functions displayed on the <Computer>, determin the value of 'a' and input it into the <Code secured door> to open it.''',
                             'Items':{},
                             'Init_position':'Puzzle room',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50 # perceptual_action: 5, # try action: 1st round: 3A, A+A, AA+A (5) pick 3A 1B 1C react A+A, AA+A, AAA+BC, B+C (9 actions)
                   },
        'puzzle3':{'Map':{'Puzzle room':[]},
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new information and keyword
                   'Space_items': {'Puzzle room':[{'Object_id':68,'Special_label':'''True,2*sin(x)'''},69]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'''You are {Name}, you need to assign values to the functions displayed on the <Computer>, determin the value of 'a' and input it into the <Code secured door> to open it.''',
                             'Items':{},
                             'Init_position':'Puzzle room',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50
                   },
        'puzzle4':{'Map':{'Puzzle room':[]},
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new information and keyword
                   'Space_items': {'Puzzle room':[{'Object_id':68,'Special_label':'''True,7*x^2'''},69]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'''You are {Name}, you need to assign values to the functions displayed on the <Computer>, determin the value of 'a' and input it into the <Code secured door> to open it.''',
                             'Items':{},
                             'Init_position':'Puzzle room',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50
                   },
        'puzzle5':{'Map':{'Puzzle room':[]},
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new information and keyword
                   'Space_items': {'Puzzle room':[{'Object_id':68,'Special_label':'''True,7*1/x'''},69]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'''You are {Name}, you need to assign values to the functions displayed on the <Computer>, determin the value of 'a' and input it into the <Code secured door> to open it.''',
                             'Items':{},
                             'Init_position':'Puzzle room',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50
                   },
        'puzzle6':{'Map':{'Puzzle room':[]},
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new information and keyword
                   'Space_items': {'Puzzle room':[{'Object_id':68,'Special_label':'''True,4*|x|,1'''},69]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'''You are {Name}. You need to assign values to the functions displayed on the <Computer>, determine the values of 'a' and 'b'. Then, input these values into the <Code secured door> in alphabetical order to open it.''',
                             'Items':{},
                             'Init_position':'Puzzle room',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50
                   },
        'puzzle7':{'Map':{'Puzzle room':[]},
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new information and keyword
                   'Space_items': {'Puzzle room':[{'Object_id':68,'Special_label':'''True,7*sin(x),2*|x|'''},69]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'''You are {Name}. You need to assign values to the functions displayed on the <Computer>, determine the values of 'a' and 'b'. Then, input these values into the <Code secured door> in alphabetical order to open it.''',
                             'Items':{},
                             'Init_position':'Puzzle room',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50
                   },
        'puzzle8':{'Map':{'Puzzle room':[]},
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new information and keyword
                   'Space_items': {'Puzzle room':[{'Object_id':68,'Special_label':'''True,3*x^2,9*sin(x)'''},69]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'''You are {Name}. You need to assign values to the functions displayed on the <Computer>, determine the values of 'a' and 'b'. Then, input these values into the <Code secured door> in alphabetical order to open it.''',
                             'Items':{},
                             'Init_position':'Puzzle room',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50
                   },
        'puzzle9':{'Map':{'Puzzle room':[]},
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new information and keyword
                   'Space_items': {'Puzzle room':[{'Object_id':68,'Special_label':'''True,9*x,4*sin(x)'''},69]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'''You are {Name}. You need to assign values to the functions displayed on the <Computer>, determine the values of 'a' and 'b'. Then, input these values into the <Code secured door> in alphabetical order to open it.''',
                             'Items':{},
                             'Init_position':'Puzzle room',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50
                    },
        'puzzle10':{'Map':{'Puzzle room':[]},
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new information and keyword
                   'Space_items': {'Puzzle room':[{'Object_id':68,'Special_label':'''True,6*x^2,4*1/x'''},69]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'''You are {Name}. You need to assign values to the functions displayed on the <Computer>, determine the values of 'a' and 'b'. Then, input these values into the <Code secured door> in alphabetical order to open it.''',
                             'Items':{},
                             'Init_position':'Puzzle room',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50
                    },
        'puzzle11':{'Map':{'Puzzle room':[]},
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new information and keyword
                   'Space_items': {'Puzzle room':[{'Object_id':68,'Special_label':'''True,6*x^2 + 4*1/x,4*1/x'''},69]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'''You are {Name}. You need to assign values to the functions displayed on the <Computer>, determine the values of 'a' and 'b'. Then, input these values into the <Code secured door> in alphabetical order to open it.''',
                             'Items':{},
                             'Init_position':'Puzzle room',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50
                    },
        'puzzle12':{'Map':{'Puzzle room':[]},
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new information and keyword
                   'Space_items': {'Puzzle room':[{'Object_id':68,'Special_label':'''True,1*sin(x) + 5*x,5*1/x, 1*|x|'''},69]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'''You are {Name}. You need to assign values to the functions displayed on the <Computer>, determine the values of 'a' and 'b'. Then, input these values into the <Code secured door> in alphabetical order to open it.''',
                             'Items':{},
                             'Init_position':'Puzzle room',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50
                    },
        'puzzle13':{'Map':{'Puzzle room':[]},
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new information and keyword
                   'Space_items': {'Puzzle room':[{'Object_id':68,'Special_label':'''True,1*sin(x)+4*1/x,4*1/x'''},69]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'''You are {Name}. You need to assign values to the functions displayed on the <Computer>, determine the values of 'a' and 'b'. Then, input these values into the <Code secured door> in alphabetical order to open it.''',
                             'Items':{},
                             'Init_position':'Puzzle room',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50
                    },
        'puzzle14':{'Map':{'Puzzle room':[]},
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new information and keyword
                   'Space_items': {'Puzzle room':[{'Object_id':68,'Special_label':'''True,2*x + 3*1/x,2*1/x'''},69]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'''You are {Name}. You need to assign values to the functions displayed on the <Computer>, determine the values of 'a' and 'b'. Then, input these values into the <Code secured door> in alphabetical order to open it.''',
                             'Items':{},
                             'Init_position':'Puzzle room',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50
                    },
        'puzzle15':{'Map':{'Puzzle room':[]},
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new information and keyword
                   'Space_items': {'Puzzle room':[{'Object_id':68,'Special_label':'''True,6*sin(x)+9*x^2,9'''},69]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'''You are {Name}. You need to assign values to the functions displayed on the <Computer>, determine the values of 'a' and 'b'. Then, input these values into the <Code secured door> in alphabetical order to open it.''',
                             'Items':{},
                             'Init_position':'Puzzle room',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50 # perceptual_action: 5, # try action: 1st round: 3A, A+A, AA+A (5) pick 3A 1B 1C react A+A, AA+A, AAA+BC, B+C (9 actions)
                   },
        'puzzle16':{'Map':{'Puzzle room':[]},
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new information and keyword
                   'Space_items': {'Puzzle room':[{'Object_id':68,'Special_label':'''True,1*sin(x)+2*1/x, 2*x^2+3*1/x,3'''},69]},# (sin(x) + 2/x, 2x^2 + 3/x, 3*x)
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'''You are {Name}. You need to assign values to the functions displayed on the <Computer>, determine the values of 'a', 'b', 'c'. Then, input these values into the <Code secured door> in alphabetical order to open it.''',
                             'Items':{},
                             'Init_position':'Puzzle room',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50
                    },
        'puzzle17':{'Map':{'Puzzle room':[]},
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new information and keyword
                   'Space_items': {'Puzzle room':[{'Object_id':68,'Special_label':'''True,9*|x|+5*x, 9*x^2+2*x,2*sin(x)'''},69]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'''You are {Name}. You need to assign values to the functions displayed on the <Computer>, determine the values of 'a', 'b', 'c'. Then, input these values into the <Code secured door> in alphabetical order to open it.''',
                             'Items':{},
                             'Init_position':'Puzzle room',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50
                    },
        'puzzle18':{'Map':{'Puzzle room':[]},
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new information and keyword
                   'Space_items': {'Puzzle room':[{'Object_id':68,'Special_label':'''True,4*sin(x)+2*1/x+3,3*x+2,4*x^2'''},69]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'''You are {Name}. You need to assign values to the functions displayed on the <Computer>, determine the values of 'a', 'b', 'c'. Then, input these values into the <Code secured door> in alphabetical order to open it.''',
                             'Items':{},
                             'Init_position':'Puzzle room',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50
                    },
        'puzzle19':{'Map':{'Puzzle room':[]},
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new information and keyword
                   'Space_items': {'Puzzle room':[{'Object_id':68,'Special_label':'''True,2*x^2+7*x+9,4*|x|+7,7*1/x'''},69]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'''You are {Name}. You need to assign values to the functions displayed on the <Computer>, determine the values of 'a', 'b', 'c', 'd'. Then, input these values into the <Code secured door> in alphabetical order to open it.''',
                             'Items':{},
                             'Init_position':'Puzzle room',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50
                    },
        'puzzle20':{'Map':{'Puzzle room':[]},
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new information and keyword
                   'Space_items': {'Puzzle room':[{'Object_id':68,'Special_label':'''True,2*x^2+7*x+9,2*x+4,9*sin(x)'''},69]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'''You are {Name}. You need to assign values to the functions displayed on the <Computer>, determine the values of 'a', 'b', 'c', 'd'. Then, input these values into the <Code secured door> in alphabetical order to open it.''',
                             'Items':{},
                             'Init_position':'Puzzle room',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50
                    },
                     }, # end of level1 puzzles
    # Level2 puzzles: clear hint of the assumption can be found in the scene, with tutorial objects 
    'Level2':{
        'puzzle1':{'Map':{'Puzzle room':[]},
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new information and keyword
                   'Space_items': {'Puzzle room':[{'Object_id':68,'Special_label':'''False,6'''},69]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'''You are {Name}, you need to assign values to the functions displayed on the <Computer>, determin the value of 'a' and input it into the <Code secured door> to open it.''',
                             'Items':{},
                             'Init_position':'Puzzle room',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50
                   },
        'puzzle2':{'Map':{'Puzzle room':[]},
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new information and keyword
                   'Space_items': {'Puzzle room':[{'Object_id':68,'Special_label':'''False,1*x'''},69]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'''You are {Name}, you need to assign values to the functions displayed on the <Computer>, determin the value of 'a' and input it into the <Code secured door> to open it.''',
                             'Items':{},
                             'Init_position':'Puzzle room',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50 # perceptual_action: 5, # try action: 1st round: 3A, A+A, AA+A (5) pick 3A 1B 1C react A+A, AA+A, AAA+BC, B+C (9 actions)
                   },
        'puzzle3':{'Map':{'Puzzle room':[]},
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new information and keyword
                   'Space_items': {'Puzzle room':[{'Object_id':68,'Special_label':'''False,2*sin(x)'''},69]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'''You are {Name}, you need to assign values to the functions displayed on the <Computer>, determin the value of 'a' and input it into the <Code secured door> to open it.''',
                             'Items':{},
                             'Init_position':'Puzzle room',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50
                   },
        'puzzle4':{'Map':{'Puzzle room':[]},
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new information and keyword
                   'Space_items': {'Puzzle room':[{'Object_id':68,'Special_label':'''False,7*x^2'''},69]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'''You are {Name}, you need to assign values to the functions displayed on the <Computer>, determin the value of 'a' and input it into the <Code secured door> to open it.''',
                             'Items':{},
                             'Init_position':'Puzzle room',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50
                   },
        'puzzle5':{'Map':{'Puzzle room':[]},
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new information and keyword
                   'Space_items': {'Puzzle room':[{'Object_id':68,'Special_label':'''False,7*1/x'''},69]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'''You are {Name}, you need to assign values to the functions displayed on the <Computer>, determin the value of 'a' and input it into the <Code secured door> to open it.''',
                             'Items':{},
                             'Init_position':'Puzzle room',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50
                   },
        'puzzle6':{'Map':{'Puzzle room':[]},
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new information and keyword
                   'Space_items': {'Puzzle room':[{'Object_id':68,'Special_label':'''False,4*|x|,1'''},69]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'''You are {Name}. You need to assign values to the functions displayed on the <Computer>, determine the values of 'a' and 'b'. Then, input these values into the <Code secured door> in alphabetical order to open it.''',
                             'Items':{},
                             'Init_position':'Puzzle room',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50
                   },
        'puzzle7':{'Map':{'Puzzle room':[]},
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new information and keyword
                   'Space_items': {'Puzzle room':[{'Object_id':68,'Special_label':'''False,7*sin(x),2*|x|'''},69]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'''You are {Name}. You need to assign values to the functions displayed on the <Computer>, determine the values of 'a' and 'b'. Then, input these values into the <Code secured door> in alphabetical order to open it.''',
                             'Items':{},
                             'Init_position':'Puzzle room',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50
                   },
        'puzzle8':{'Map':{'Puzzle room':[]},
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new information and keyword
                   'Space_items': {'Puzzle room':[{'Object_id':68,'Special_label':'''False,3*x^2,9*sin(x)'''},69]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'''You are {Name}. You need to assign values to the functions displayed on the <Computer>, determine the values of 'a' and 'b'. Then, input these values into the <Code secured door> in alphabetical order to open it.''',
                             'Items':{},
                             'Init_position':'Puzzle room',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50
                   },
        'puzzle9':{'Map':{'Puzzle room':[]},
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new information and keyword
                   'Space_items': {'Puzzle room':[{'Object_id':68,'Special_label':'''False,9*x,4*sin(x)'''},69]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'''You are {Name}. You need to assign values to the functions displayed on the <Computer>, determine the values of 'a' and 'b'. Then, input these values into the <Code secured door> in alphabetical order to open it.''',
                             'Items':{},
                             'Init_position':'Puzzle room',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50
                    },
        'puzzle10':{'Map':{'Puzzle room':[]},
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new information and keyword
                   'Space_items': {'Puzzle room':[{'Object_id':68,'Special_label':'''False,6*x^2,4*1/x'''},69]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'''You are {Name}. You need to assign values to the functions displayed on the <Computer>, determine the values of 'a' and 'b'. Then, input these values into the <Code secured door> in alphabetical order to open it.''',
                             'Items':{},
                             'Init_position':'Puzzle room',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50
                    },
        'puzzle11':{'Map':{'Puzzle room':[]},
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new information and keyword
                   'Space_items': {'Puzzle room':[{'Object_id':68,'Special_label':'''False,6*x^2 + 4*1/x,4*1/x'''},69]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'''You are {Name}. You need to assign values to the functions displayed on the <Computer>, determine the values of 'a' and 'b'. Then, input these values into the <Code secured door> in alphabetical order to open it.''',
                             'Items':{},
                             'Init_position':'Puzzle room',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50
                    },
        'puzzle12':{'Map':{'Puzzle room':[]},
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new information and keyword
                   'Space_items': {'Puzzle room':[{'Object_id':68,'Special_label':'''False,1*sin(x) + 5*x,5*1/x, 1*|x|'''},69]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'''You are {Name}. You need to assign values to the functions displayed on the <Computer>, determine the values of 'a' and 'b'. Then, input these values into the <Code secured door> in alphabetical order to open it.''',
                             'Items':{},
                             'Init_position':'Puzzle room',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50
                    },
        'puzzle13':{'Map':{'Puzzle room':[]},
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new information and keyword
                   'Space_items': {'Puzzle room':[{'Object_id':68,'Special_label':'''False,1*sin(x)+4*1/x,4*1/x'''},69]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'''You are {Name}. You need to assign values to the functions displayed on the <Computer>, determine the values of 'a' and 'b'. Then, input these values into the <Code secured door> in alphabetical order to open it.''',
                             'Items':{},
                             'Init_position':'Puzzle room',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50
                    },
        'puzzle14':{'Map':{'Puzzle room':[]},
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new information and keyword
                   'Space_items': {'Puzzle room':[{'Object_id':68,'Special_label':'''False,2*x + 3*1/x,2*1/x'''},69]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'''You are {Name}. You need to assign values to the functions displayed on the <Computer>, determine the values of 'a' and 'b'. Then, input these values into the <Code secured door> in alphabetical order to open it.''',
                             'Items':{},
                             'Init_position':'Puzzle room',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50
                    },
        'puzzle15':{'Map':{'Puzzle room':[]},
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new information and keyword
                   'Space_items': {'Puzzle room':[{'Object_id':68,'Special_label':'''False,6*sin(x)+9*x^2,9'''},69]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'''You are {Name}. You need to assign values to the functions displayed on the <Computer>, determine the values of 'a' and 'b'. Then, input these values into the <Code secured door> in alphabetical order to open it.''',
                             'Items':{},
                             'Init_position':'Puzzle room',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50 # perceptual_action: 5, # try action: 1st round: 3A, A+A, AA+A (5) pick 3A 1B 1C react A+A, AA+A, AAA+BC, B+C (9 actions)
                   },
        'puzzle16':{'Map':{'Puzzle room':[]},
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new information and keyword
                   'Space_items': {'Puzzle room':[{'Object_id':68,'Special_label':'''False,1*sin(x)+2*1/x, 2*x^2+3*1/x,3'''},69]},# (sin(x) + 2/x, 2x^2 + 3/x, 3*x)
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'''You are {Name}. You need to assign values to the functions displayed on the <Computer>, determine the values of 'a', 'b', 'c'. Then, input these values into the <Code secured door> in alphabetical order to open it.''',
                             'Items':{},
                             'Init_position':'Puzzle room',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50
                    },
        'puzzle17':{'Map':{'Puzzle room':[]},
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new information and keyword
                   'Space_items': {'Puzzle room':[{'Object_id':68,'Special_label':'''False,9*|x|+5*x, 9*x^2+2*x,2*sin(x)'''},69]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'''You are {Name}. You need to assign values to the functions displayed on the <Computer>, determine the values of 'a', 'b', 'c'. Then, input these values into the <Code secured door> in alphabetical order to open it.''',
                             'Items':{},
                             'Init_position':'Puzzle room',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50
                    },
        'puzzle18':{'Map':{'Puzzle room':[]},
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new information and keyword
                   'Space_items': {'Puzzle room':[{'Object_id':68,'Special_label':'''False,4*sin(x)+2*1/x+3,3*x+2,4*x^2'''},69]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'''You are {Name}. You need to assign values to the functions displayed on the <Computer>, determine the values of 'a', 'b', 'c'. Then, input these values into the <Code secured door> in alphabetical order to open it.''',
                             'Items':{},
                             'Init_position':'Puzzle room',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50
                    },
        'puzzle19':{'Map':{'Puzzle room':[]},
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new information and keyword
                   'Space_items': {'Puzzle room':[{'Object_id':68,'Special_label':'''False,2*x^2+7*x+9,4*|x|+7,7*1/x'''},69]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'''You are {Name}. You need to assign values to the functions displayed on the <Computer>, determine the values of 'a', 'b', 'c', 'd'. Then, input these values into the <Code secured door> in alphabetical order to open it.''',
                             'Items':{},
                             'Init_position':'Puzzle room',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50
                    },
        'puzzle20':{'Map':{'Puzzle room':[]},
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new information and keyword
                   'Space_items': {'Puzzle room':[{'Object_id':68,'Special_label':'''False,2*x^2+7*x+9,2*x+4,9*sin(x)'''},69]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'''You are {Name}. You need to assign values to the functions displayed on the <Computer>, determine the values of 'a', 'b', 'c', 'd'. Then, input these values into the <Code secured door> in alphabetical order to open it.''',
                             'Items':{},
                             'Init_position':'Puzzle room',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50
                    },
                     }, # end of level2 puzzles 
} # end of simple gallary puzzle

# --------------------------------------Gallery Puzzle--------------------------------------
# --------------------------------------Gallery Puzzle--------------------------------------
# --------------------------------------Gallery Puzzle--------------------------------------

# oil paints
# 44:Blue, 45:Yellow, 46:Black, 47:Yellow, 48:Grreen, 49:Blue, 50:Blue, 51:Blue,

# acrylic paints
# 52:Black, 53:Yellow, 54:Green, 55:Green, 56:Blue, 57:Blue

# watercolor paints
# 58:Yellow, 59:Green, 60:Blue, 61:Blue, 62:Blue, 
Art_gallery_puzzles = {
    # Level1 puzzles, clear assumption can be found in the scene，Only deduction is evaluated in this step
    'Level1':{
        'puzzle1':{'Map':{'Corridor':[]},
                   # oil paints
                    # 44:Blue, 45:Yellow, 46:Black, 47:Yellow, 48:Grreen, 49:Blue, 50:Blue, 51:Blue,
                    
                    # acrylic paints
                    # 52:Black, 53:Yellow, 54:Green, 55:Green, 56:Blue, 57:Blue
                    
                    # watercolor paints
                    # 58:Yellow, 59:Green, 60:Blue, 61:Blue, 62:Blue, 
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new Keyword and information
                   'Space_items': {'Corridor':[{'Object_id':65,'Special_label':'Level1'},50,56,59]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'You are {Name} ', # hard coded in the run_experiment.py
                             'Items':{},
                             'Init_position':'Corridor',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50, # perceptual_action: 14， try_action:2+3edge
                   },
        'puzzle2':{'Map':{'Corridor':[]},
                   # oil paints
                    # 44:Blue, 45:Yellow, 46:Black, 47:Yellow, 48:Grreen, 49:Blue, 50:Blue, 51:Blue,
                    
                    # acrylic paints
                    # 52:Black, 53:Yellow, 54:Green, 55:Green, 56:Blue, 57:Blue
                    
                    # watercolor paints
                    # 58:Yellow, 59:Green, 60:Blue, 61:Blue, 62:Blue, 
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new Keyword and information
                   'Space_items': {'Corridor':[{'Object_id':65,'Special_label':'Level1'},50,47,54,58,61]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'You are {Name} and you are now playing an escape room game where you need to escape this place.',
                             'Items':{},
                             'Init_position':'Corridor',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50, # perceptual_action: 14， try_action:2+3edge
                   },
        'puzzle3':{'Map':{'Corridor':[]},
                   # oil paints
                    # 44:Blue, 45:Yellow, 46:Black, 47:Yellow, 48:Grreen, 49:Blue, 50:Blue, 51:Blue,
                    
                    # acrylic paints
                    # 52:Black, 53:Yellow, 54:Green, 55:Green, 56:Blue, 57:Blue
                    
                    # watercolor paints
                    # 58:Yellow, 59:Green, 60:Blue, 61:Blue, 62:Blue, 
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new Keyword and information
                   'Space_items': {'Corridor':[{'Object_id':65,'Special_label':'Level1'},44,50,54,57,58,61,59]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'You are {Name} and you are now playing an escape room game where you need to escape this place.',
                             'Items':{},
                             'Init_position':'Corridor',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50, # perceptual_action: 14， try_action:2+3edge
                   },
        'puzzle4':{'Map':{'Corridor':[]},
                   # oil paints
                    # 44:Blue, 45:Yellow, 46:Black, 47:Yellow, 48:Grreen, 49:Blue, 50:Blue, 51:Blue,
                    
                    # acrylic paints
                    # 52:Black, 53:Yellow, 54:Green, 55:Green, 56:Blue, 57:Blue
                    
                    # watercolor paints
                    # 58:Yellow, 59:Green, 60:Blue, 61:Blue, 62:Blue, 
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new Keyword and information
                   'Space_items': {'Corridor':[{'Object_id':65,'Special_label':'Level1'},45,47,51,49,55,56,62]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'You are {Name} and you are now playing an escape room game where you need to escape this place.',
                             'Items':{},
                             'Init_position':'Corridor',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50, # perceptual_action: 14， try_action:2+3edge
                   },
        'puzzle5':{'Map':{'Corridor':[]},
                   # oil paints
                    # 44:Blue, 45:Yellow, 46:Black, 47:Yellow, 48:Grreen, 49:Blue, 50:Blue, 51:Blue,
                    
                    # acrylic paints
                    # 52:Black, 53:Yellow, 54:Green, 55:Green, 56:Blue, 57:Blue
                    
                    # watercolor paints
                    # 58:Yellow, 59:Green, 60:Blue, 61:Blue, 62:Blue, 
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new Keyword and information
                   'Space_items': {'Corridor':[{'Object_id':65,'Special_label':'Level1'},50,54,57,55,52,62,60,61]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'You are {Name} and you are now playing an escape room game where you need to escape this place.',
                             'Items':{},
                             'Init_position':'Corridor',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50, # perceptual_action: 14， try_action:2+3edge
                   },
        'puzzle6':{'Map':{'Corridor':[]},
                   # oil paints
                    # 44:Blue, 45:Yellow, 46:Black, 47:Yellow, 48:Grreen, 49:Blue, 50:Blue, 51:Blue,
                    
                    # acrylic paints
                    # 52:Black, 53:Yellow, 54:Green, 55:Green, 56:Blue, 57:Blue
                    
                    # watercolor paints
                    # 58:Yellow, 59:Green, 60:Blue, 61:Blue, 62:Blue, 
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new Keyword and information
                   'Space_items': {'Corridor':[{'Object_id':65,'Special_label':'Level1'},48,44,49,53,56,57,60,62,59]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'You are {Name} and you are now playing an escape room game where you need to escape this place.',
                             'Items':{},
                             'Init_position':'Corridor',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50, # perceptual_action: 14， try_action:2+3edge
                   },
        'puzzle7':{'Map':{'Corridor':[]},
                   # oil paints
                    # 44:Blue, 45:Yellow, 46:Black, 47:Yellow, 48:Grreen, 49:Blue, 50:Blue, 51:Blue,
                    
                    # acrylic paints
                    # 52:Black, 53:Yellow, 54:Green, 55:Green, 56:Blue, 57:Blue
                    
                    # watercolor paints
                    # 58:Yellow, 59:Green, 60:Blue, 61:Blue, 62:Blue, 
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new Keyword and information
                   'Space_items': {'Corridor':[{'Object_id':65,'Special_label':'Level1'},45,49,51,56,52,55,53,58,61,62]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'You are {Name} and you are now playing an escape room game where you need to escape this place.',
                             'Items':{},
                             'Init_position':'Corridor',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50, # perceptual_action: 14， try_action:2+3edge
                   },
        'puzzle8':{'Map':{'Corridor':[]},
                   # oil paints
                    # 44:Blue, 45:Yellow, 46:Black, 47:Yellow, 48:Grreen, 49:Blue, 50:Blue, 51:Blue,
                    
                    # acrylic paints
                    # 52:Black, 53:Yellow, 54:Green, 55:Green, 56:Blue, 57:Blue
                    
                    # watercolor paints
                    # 58:Yellow, 59:Green, 60:Blue, 61:Blue, 62:Blue, 
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new Keyword and information
                   'Space_items': {'Corridor':[{'Object_id':65,'Special_label':'Level1'},49,45,47,53,55,57,54,56,58,62,60]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'You are {Name} and you are now playing an escape room game where you need to escape this place.',
                             'Items':{},
                             'Init_position':'Corridor',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50, # perceptual_action: 14， try_action:2+3edge
                   },
        'puzzle9':{'Map':{'Corridor':[]},
                   # oil paints
                    # 44:Blue, 45:Yellow, 46:Black, 47:Yellow, 48:Grreen, 49:Blue, 50:Blue, 51:Blue,
                    
                    # acrylic paints
                    # 52:Black, 53:Yellow, 54:Green, 55:Green, 56:Blue, 57:Blue
                    
                    # watercolor paints
                    # 58:Yellow, 59:Green, 60:Blue, 61:Blue, 62:Blue, 
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new Keyword and information
                   'Space_items': {'Corridor':[{'Object_id':65,'Special_label':'Level1'},49,50,46,44,53,56,57,52,55,61,58,62]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'You are {Name} and you are now playing an escape room game where you need to escape this Place.',
                             'Items':{},
                             'Init_position':'Corridor',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50, # perceptual_action: 14， try_action:2+3edge
                   },
        'puzzle10':{'Map':{'Corridor':[]},
                   # oil paints
                    # 44:Blue, 45:Yellow, 46:Black, 47:Yellow, 48:Grreen, 49:Blue, 50:Blue, 51:Blue,
                    
                    # acrylic paints
                    # 52:Black, 53:Yellow, 54:Green, 55:Green, 56:Blue, 57:Blue
                    
                    # watercolor paints
                    # 58:Yellow, 59:Green, 60:Blue, 61:Blue, 62:Blue, 
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new Keyword and information
                   'Space_items': {'Corridor':[{'Object_id':65,'Special_label':'Level1'},48,50,51,47,44,53,52,57,59,61,60,58,62]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'You are {Name} and you are now playing an escape room game where you need to escape this place.',
                             'Items':{},
                             'Init_position':'Corridor',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50, # perceptual_action: 14， try_action:2+3edge
                    },
        'puzzle11':{'Map':{
                          'Oil Painting Gallery':['Watercolour Gallery','Acrylic Painting Gallery'],
                          'Acrylic Painting Gallery':['Watercolour Gallery'],
                          'Watercolour Gallery':['Acrylic Painting Gallery']},
                    # oil paints
                    # 44:Blue, 45:Yellow, 46:Black, 47:Yellow, 48:Grreen, 49:Blue, 50:Blue, 51:Blue,
                    
                    # acrylic paints
                    # 52:Black, 53:Yellow, 54:Green, 55:Green, 56:Blue, 57:Blue
                    
                    # watercolor paints
                    # 58:Yellow, 59:Green, 60:Blue, 61:Blue, 62:Blue, 
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new Keyword and information
                   'Space_items': {
                                   'Oil Painting Gallery':[50,{'Object_id':65,'Special_label':'Level1'}],
                                   'Acrylic Painting Gallery':[56],
                                   'Watercolour Gallery':[59]},
                   'Space_item_containers':{},
                   'Edges': {'Oil Painting Gallery':{ 'Watercolour Gallery':[31], 'Acrylic Painting Gallery':[30]}},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'You are {Name} and you are now playing an escape room game where you need to escape this place.',
                             'Items':{},
                             'Init_position':'Oil Painting Gallery',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50 # perceptual_action: 11, # try action: 1+3
                    },
        'puzzle12':{'Map':{
                          'Oil Painting Gallery':['Watercolour Gallery','Acrylic Painting Gallery'],
                          'Acrylic Painting Gallery':['Watercolour Gallery'],
                          'Watercolour Gallery':['Acrylic Painting Gallery']},
                    # oil paints
                    # 44:Blue, 45:Yellow, 46:Black, 47:Yellow, 48:Grreen, 49:Blue, 50:Blue, 51:Blue,
                    
                    # acrylic paints
                    # 52:Black, 53:Yellow, 54:Green, 55:Green, 56:Blue, 57:Blue
                    
                    # watercolor paints
                    # 58:Yellow, 59:Green, 60:Blue, 61:Blue, 62:Blue, 
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new Keyword and information
                   'Space_items': {
                                   'Oil Painting Gallery':[50,47,{'Object_id':65,'Special_label':'Level1'}],
                                   'Acrylic Painting Gallery':[54],
                                   'Watercolour Gallery':[58,61]},
                   'Space_item_containers':{},
                   'Edges': {'Oil Painting Gallery':{ 'Watercolour Gallery':[31], 'Acrylic Painting Gallery':[30]}},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'You are {Name} and you are now playing an escape room game where you need to escape this place.',
                             'Items':{},
                             'Init_position':'Oil Painting Gallery',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50 # perceptual_action: 11, # try action: 1+3
                    },
        'puzzle13':{'Map':{
                          'Oil Painting Gallery':['Watercolour Gallery','Acrylic Painting Gallery'],
                          'Acrylic Painting Gallery':['Watercolour Gallery'],
                          'Watercolour Gallery':['Acrylic Painting Gallery']},
                    # oil paints
                    # 44:Blue, 45:Yellow, 46:Black, 47:Yellow, 48:Grreen, 49:Blue, 50:Blue, 51:Blue,
                    
                    # acrylic paints
                    # 52:Black, 53:Yellow, 54:Green, 55:Green, 56:Blue, 57:Blue
                    
                    # watercolor paints
                    # 58:Yellow, 59:Green, 60:Blue, 61:Blue, 62:Blue, 
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new Keyword and information
                   'Space_items': {
                                   'Oil Painting Gallery':[48,49,{'Object_id':65,'Special_label':'Level1'}],
                                   'Acrylic Painting Gallery':[53,57],
                                   'Watercolour Gallery':[62,61]},
                   'Space_item_containers':{},
                   'Edges': {'Oil Painting Gallery':{'Watercolour Gallery':[31], 'Acrylic Painting Gallery':[30]}},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'You are {Name} and you are now playing an escape room game where you need to escape this place.',
                             'Items':{},
                             'Init_position':'Oil Painting Gallery',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50 # perceptual_action: 11, # try action: 1+3
                    },
        'puzzle14':{'Map':{
                          'Oil Painting Gallery':['Watercolour Gallery','Acrylic Painting Gallery'],
                          'Acrylic Painting Gallery':['Watercolour Gallery'],
                          'Watercolour Gallery':['Acrylic Painting Gallery']},
                    # oil paints
                    # 44:Blue, 45:Yellow, 46:Black, 47:Yellow, 48:Grreen, 49:Blue, 50:Blue, 51:Blue,
                    
                    # acrylic paints
                    # 52:Black, 53:Yellow, 54:Green, 55:Green, 56:Blue, 57:Blue
                    
                    # watercolor paints
                    # 58:Yellow, 59:Green, 60:Blue, 61:Blue, 62:Blue, 
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new Keyword and information
                   'Space_items': {
                                   'Oil Painting Gallery':[45, 47, 51, 49,{'Object_id':65,'Special_label':'Level1'}],
                                   'Acrylic Painting Gallery':[55,56],
                                   'Watercolour Gallery':[62]},
                   'Space_item_containers':{},
                   'Edges': {'Oil Painting Gallery':{'Watercolour Gallery':[31], 'Acrylic Painting Gallery':[30]}},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'You are {Name} and you are now playing an escape room game where you need to escape this place.',
                             'Items':{},
                             'Init_position':'Oil Painting Gallery',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50 # perceptual_action: 11, # try action: 1+3
                    },
        'puzzle15':{'Map':{
                          'Oil Painting Gallery':['Watercolour Gallery','Acrylic Painting Gallery'],
                          'Acrylic Painting Gallery':['Watercolour Gallery'],
                          'Watercolour Gallery':['Acrylic Painting Gallery']},
                    # oil paints
                    # 44:Blue, 45:Yellow, 46:Black, 47:Yellow, 48:Grreen, 49:Blue, 50:Blue, 51:Blue,
                    
                    # acrylic paints
                    # 52:Black, 53:Yellow, 54:Green, 55:Green, 56:Blue, 57:Blue
                    
                    # watercolor paints
                    # 58:Yellow, 59:Green, 60:Blue, 61:Blue, 62:Blue, 
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new Keyword and information
                   'Space_items': {
                                   'Oil Painting Gallery':[50,{'Object_id':65,'Special_label':'Level1'}],
                                   'Acrylic Painting Gallery':[54, 57, 55, 52],
                                   'Watercolour Gallery':[62, 60, 61]},
                   'Space_item_containers':{},
                   'Edges': {'Oil Painting Gallery':{'Watercolour Gallery':[31], 'Acrylic Painting Gallery':[30]}},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'You are {Name} and you are now playing an escape room game where you need to escape this place.',
                             'Items':{},
                             'Init_position':'Oil Painting Gallery',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50 # perceptual_action: 11, # try action: 1+3
                    },
        'puzzle16':{'Map':{
                          'Oil Painting Gallery':['Watercolour Gallery','Acrylic Painting Gallery'],
                          'Acrylic Painting Gallery':['Watercolour Gallery'],
                          'Watercolour Gallery':['Acrylic Painting Gallery']},
                    # oil paints
                    # 44:Blue, 45:Yellow, 46:Black, 47:Yellow, 48:Grreen, 49:Blue, 50:Blue, 51:Blue,
                    
                    # acrylic paints
                    # 52:Black, 53:Yellow, 54:Green, 55:Green, 56:Blue, 57:Blue
                    
                    # watercolor paints
                    # 58:Yellow, 59:Green, 60:Blue, 61:Blue, 62:Blue, 
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new Keyword and information
                   'Space_items': {
                                   'Oil Painting Gallery':[48, 44, 49, {'Object_id':65,'Special_label':'Level1'}],
                                   'Acrylic Painting Gallery':[53, 56, 57],
                                   'Watercolour Gallery':[60, 62, 59]},
                   'Space_item_containers':{},
                   'Edges': {'Oil Painting Gallery':{'Watercolour Gallery':[31], 'Acrylic Painting Gallery':[30]}},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'You are {Name} and you are now playing an escape room game where you need to escape this place.',
                             'Items':{},
                             'Init_position':'Oil Painting Gallery',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50 # perceptual_action: 11, # try action: 1+3
                    },
        'puzzle17':{'Map':{
                          'Oil Painting Gallery':['Watercolour Gallery','Acrylic Painting Gallery'],
                          'Acrylic Painting Gallery':['Watercolour Gallery'],
                          'Watercolour Gallery':['Acrylic Painting Gallery']},
                    # oil paints
                    # 44:Blue, 45:Yellow, 46:Black, 47:Yellow, 48:Grreen, 49:Blue, 50:Blue, 51:Blue,
                    
                    # acrylic paints
                    # 52:Black, 53:Yellow, 54:Green, 55:Green, 56:Blue, 57:Blue
                    
                    # watercolor paints
                    # 58:Yellow, 59:Green, 60:Blue, 61:Blue, 62:Blue, 
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new Keyword and information
                   'Space_items': {
                                   'Oil Painting Gallery':[45, 49, 51,{'Object_id':65,'Special_label':'Level1'}],
                                   'Acrylic Painting Gallery':[56, 52, 55, 53],
                                   'Watercolour Gallery':[58, 61, 62]},
                   'Space_item_containers':{},
                   'Edges': {'Oil Painting Gallery':{'Watercolour Gallery':[31], 'Acrylic Painting Gallery':[30]}},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'You are {Name} and you are now playing an escape room game where you need to escape this place.',
                             'Items':{},
                             'Init_position':'Oil Painting Gallery',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50 # perceptual_action: 11, # try action: 1+3
                    },
        'puzzle18':{'Map':{
                          'Oil Painting Gallery':['Watercolour Gallery','Acrylic Painting Gallery'],
                          'Acrylic Painting Gallery':['Watercolour Gallery'],
                          'Watercolour Gallery':['Acrylic Painting Gallery']},
                    # oil paints
                    # 44:Blue, 45:Yellow, 46:Black, 47:Yellow, 48:Grreen, 49:Blue, 50:Blue, 51:Blue,
                    
                    # acrylic paints
                    # 52:Black, 53:Yellow, 54:Green, 55:Green, 56:Blue, 57:Blue
                    
                    # watercolor paints
                    # 58:Yellow, 59:Green, 60:Blue, 61:Blue, 62:Blue, 
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new Keyword and information
                   'Space_items': {
                                   'Oil Painting Gallery':[49, 45, 47,{'Object_id':65,'Special_label':'Level1'}],
                                   'Acrylic Painting Gallery':[53, 55, 57, 54, 56],
                                   'Watercolour Gallery':[58, 62, 60]},
                   'Space_item_containers':{},
                   'Edges': {'Oil Painting Gallery':{'Watercolour Gallery':[31], 'Acrylic Painting Gallery':[30]}},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'You are {Name} and you are now playing an escape room game where you need to escape this place.',
                             'Items':{},
                             'Init_position':'Oil Painting Gallery',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50 # perceptual_action: 11, # try action: 1+3
                    },
        'puzzle19':{'Map':{
                          'Oil Painting Gallery':['Watercolour Gallery','Acrylic Painting Gallery'],
                          'Acrylic Painting Gallery':['Watercolour Gallery'],
                          'Watercolour Gallery':['Acrylic Painting Gallery']},
                    # oil paints
                    # 44:Blue, 45:Yellow, 46:Black, 47:Yellow, 48:Grreen, 49:Blue, 50:Blue, 51:Blue,
                    
                    # acrylic paints
                    # 52:Black, 53:Yellow, 54:Green, 55:Green, 56:Blue, 57:Blue
                    
                    # watercolor paints
                    # 58:Yellow, 59:Green, 60:Blue, 61:Blue, 62:Blue, 
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new Keyword and information
                   'Space_items': {
                                   'Oil Painting Gallery':[49, 50, 46, 44,{'Object_id':65,'Special_label':'Level1'}],
                                   'Acrylic Painting Gallery':[53, 56, 57, 52, 55],
                                   'Watercolour Gallery':[61, 58, 62]},
                   'Space_item_containers':{},
                   'Edges': {'Oil Painting Gallery':{'Watercolour Gallery':[31], 'Acrylic Painting Gallery':[30]}},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'You are {Name} and you are now playing an escape room game where you need to escape this place.',
                             'Items':{},
                             'Init_position':'Oil Painting Gallery',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50 # perceptual_action: 11, # try action: 1+3
                    },
        'puzzle20':{'Map':{
                          'Oil Painting Gallery':['Watercolour Gallery','Acrylic Painting Gallery'],
                          'Acrylic Painting Gallery':['Watercolour Gallery'],
                          'Watercolour Gallery':['Acrylic Painting Gallery']},
                    # oil paints
                    # 44:Blue, 45:Yellow, 46:Black, 47:Yellow, 48:Grreen, 49:Blue, 50:Blue, 51:Blue,
                    
                    # acrylic paints
                    # 52:Black, 53:Yellow, 54:Green, 55:Green, 56:Blue, 57:Blue
                    
                    # watercolor paints
                    # 58:Yellow, 59:Green, 60:Blue, 61:Blue, 62:Blue, 
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new Keyword and information
                   'Space_items': {
                                   'Oil Painting Gallery':[48, 50, 51, 47, 44,{'Object_id':65,'Special_label':'Level1'}],
                                   'Acrylic Painting Gallery':[53, 52, 57],
                                   'Watercolour Gallery':[59, 61, 60, 58, 62]},
                   'Space_item_containers':{},
                   'Edges': {'Oil Painting Gallery':{'Watercolour Gallery':[31], 'Acrylic Painting Gallery':[30]}},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'You are {Name} and you are now playing an escape room game where you need to escape this place.',
                             'Items':{},
                             'Init_position':'Oil Painting Gallery',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50 # perceptual_action: 11, # try action: 1+3
                    },
                     }, # end of level1 puzzles
    # Level2 puzzles: clear hint of the assumption can be found in the scene, with tutorial objects 
    'Level2':{
        
        'puzzle1':{'Map':{'Corridor':[]},
                   # oil paints
                    # 44:Blue, 45:Yellow, 46:Black, 47:Yellow, 48:Grreen, 49:Blue, 50:Blue, 51:Blue,
                    
                    # acrylic paints
                    # 52:Black, 53:Yellow, 54:Green, 55:Green, 56:Blue, 57:Blue
                    
                    # watercolor paints
                    # 58:Yellow, 59:Green, 60:Blue, 61:Blue, 62:Blue, 
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new Keyword and information
                   'Space_items': {'Corridor':[{'Object_id':65,'Special_label':'Level2'},50,56,59]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'You are {Name} and you are now playing an escape room game where you need to escape this place.',
                             'Items':{},
                             'Init_position':'Corridor',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50, # perceptual_action: 14， try_action:2+3edge
                   },
        'puzzle2':{'Map':{'Corridor':[]},
                   # oil paints
                    # 44:Blue, 45:Yellow, 46:Black, 47:Yellow, 48:Grreen, 49:Blue, 50:Blue, 51:Blue,
                    
                    # acrylic paints
                    # 52:Black, 53:Yellow, 54:Green, 55:Green, 56:Blue, 57:Blue
                    
                    # watercolor paints
                    # 58:Yellow, 59:Green, 60:Blue, 61:Blue, 62:Blue, 
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new Keyword and information
                   'Space_items': {'Corridor':[{'Object_id':65,'Special_label':'Level2'},50,47,54,58,61]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'You are {Name} and you are now playing an escape room game where you need to escape this place.',
                             'Items':{},
                             'Init_position':'Corridor',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50, # perceptual_action: 14， try_action:2+3edge
                   },
        'puzzle3':{'Map':{'Corridor':[]},
                   # oil paints
                    # 44:Blue, 45:Yellow, 46:Black, 47:Yellow, 48:Grreen, 49:Blue, 50:Blue, 51:Blue,
                    
                    # acrylic paints
                    # 52:Black, 53:Yellow, 54:Green, 55:Green, 56:Blue, 57:Blue
                    
                    # watercolor paints
                    # 58:Yellow, 59:Green, 60:Blue, 61:Blue, 62:Blue, 
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new Keyword and information
                   'Space_items': {'Corridor':[{'Object_id':65,'Special_label':'Level2'},48,49,57,53,62,61]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'You are {Name} and you are now playing an escape room game where you need to escape this place.',
                             'Items':{},
                             'Init_position':'Corridor',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50, # perceptual_action: 14， try_action:2+3edge
                   },
        'puzzle4':{'Map':{'Corridor':[]},
                   # oil paints
                    # 44:Blue, 45:Yellow, 46:Black, 47:Yellow, 48:Grreen, 49:Blue, 50:Blue, 51:Blue,
                    
                    # acrylic paints
                    # 52:Black, 53:Yellow, 54:Green, 55:Green, 56:Blue, 57:Blue
                    
                    # watercolor paints
                    # 58:Yellow, 59:Green, 60:Blue, 61:Blue, 62:Blue, 
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new Keyword and information
                   'Space_items': {'Corridor':[{'Object_id':65,'Special_label':'Level2'},45,47,51,49,55,56,62]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'You are {Name} and you are now playing an escape room game where you need to escape this place.',
                             'Items':{},
                             'Init_position':'Corridor',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50, # perceptual_action: 14， try_action:2+3edge
                   },
        'puzzle5':{'Map':{'Corridor':[]},
                   # oil paints
                    # 44:Blue, 45:Yellow, 46:Black, 47:Yellow, 48:Grreen, 49:Blue, 50:Blue, 51:Blue,
                    
                    # acrylic paints
                    # 52:Black, 53:Yellow, 54:Green, 55:Green, 56:Blue, 57:Blue
                    
                    # watercolor paints
                    # 58:Yellow, 59:Green, 60:Blue, 61:Blue, 62:Blue, 
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new Keyword and information
                   'Space_items': {'Corridor':[{'Object_id':65,'Special_label':'Level2'},60,54,57,55,52,62,60,61]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'You are {Name} and you are now playing an escape room game where you need to escape this place.',
                             'Items':{},
                             'Init_position':'Corridor',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50, # perceptual_action: 14， try_action:2+3edge
                   },
        'puzzle6':{'Map':{'Corridor':[]},
                   # oil paints
                    # 44:Blue, 45:Yellow, 46:Black, 47:Yellow, 48:Grreen, 49:Blue, 50:Blue, 51:Blue,
                    
                    # acrylic paints
                    # 52:Black, 53:Yellow, 54:Green, 55:Green, 56:Blue, 57:Blue
                    
                    # watercolor paints
                    # 58:Yellow, 59:Green, 60:Blue, 61:Blue, 62:Blue, 
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new Keyword and information
                   'Space_items': {'Corridor':[{'Object_id':65,'Special_label':'Level2'},48,44,49,53,56,57,60,62,59]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'You are {Name} and you are now playing an escape room game where you need to escape this place.',
                             'Items':{},
                             'Init_position':'Corridor',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50, # perceptual_action: 14， try_action:2+3edge
                   },
        'puzzle7':{'Map':{'Corridor':[]},
                   # oil paints
                    # 44:Blue, 45:Yellow, 46:Black, 47:Yellow, 48:Grreen, 49:Blue, 50:Blue, 51:Blue,
                    
                    # acrylic paints
                    # 52:Black, 53:Yellow, 54:Green, 55:Green, 56:Blue, 57:Blue
                    
                    # watercolor paints
                    # 58:Yellow, 59:Green, 60:Blue, 61:Blue, 62:Blue, 
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new Keyword and information
                   'Space_items': {'Corridor':[{'Object_id':65,'Special_label':'Level2'},45,49,51,56,52,55,53]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'You are {Name} and you are now playing an escape room game where you need to escape this place.',
                             'Items':{},
                             'Init_position':'Corridor',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50, # perceptual_action: 14， try_action:2+3edge
                   },
        'puzzle8':{'Map':{'Corridor':[]},
                   # oil paints
                    # 44:Blue, 45:Yellow, 46:Black, 47:Yellow, 48:Grreen, 49:Blue, 50:Blue, 51:Blue,
                    
                    # acrylic paints
                    # 52:Black, 53:Yellow, 54:Green, 55:Green, 56:Blue, 57:Blue
                    
                    # watercolor paints
                    # 58:Yellow, 59:Green, 60:Blue, 61:Blue, 62:Blue, 
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new Keyword and information
                   'Space_items': {'Corridor':[{'Object_id':65,'Special_label':'Level2'},49,45,47,53,55,57,54,56,58,62,60]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'You are {Name} and you are now playing an escape room game where you need to escape this place.',
                             'Items':{},
                             'Init_position':'Corridor',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50, # perceptual_action: 14， try_action:2+3edge
                   },
        'puzzle9':{'Map':{'Corridor':[]},
                   # oil paints
                    # 44:Blue, 45:Yellow, 46:Black, 47:Yellow, 48:Grreen, 49:Blue, 50:Blue, 51:Blue,
                    
                    # acrylic paints
                    # 52:Black, 53:Yellow, 54:Green, 55:Green, 56:Blue, 57:Blue
                    
                    # watercolor paints
                    # 58:Yellow, 59:Green, 60:Blue, 61:Blue, 62:Blue, 
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new Keyword and information
                   'Space_items': {'Corridor':[{'Object_id':65,'Special_label':'Level2'},49,50,46,44,53,56,57,52,55,61,58,62]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'You are {Name} and you are now playing an escape room game where you need to escape this place.',
                             'Items':{},
                             'Init_position':'Corridor',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50, # perceptual_action: 14， try_action:2+3edge
                   },
        'puzzle10':{'Map':{'Corridor':[]},
                   # oil paints
                    # 44:Blue, 45:Yellow, 46:Black, 47:Yellow, 48:Grreen, 49:Blue, 50:Blue, 51:Blue,
                    
                    # acrylic paints
                    # 52:Black, 53:Yellow, 54:Green, 55:Green, 56:Blue, 57:Blue
                    
                    # watercolor paints
                    # 58:Yellow, 59:Green, 60:Blue, 61:Blue, 62:Blue, 
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new Keyword and information
                   'Space_items': {'Corridor':[{'Object_id':65,'Special_label':'Level2'},48,50,46,44,53,56,57,52,55,61,58,62]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'You are {Name} and you are now playing an escape room game where you need to escape this place.',
                             'Items':{},
                             'Init_position':'Corridor',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50, # perceptual_action: 14， try_action:2+3edge
                    },
        'puzzle11':{'Map':{
                          'Oil Painting Gallery':['Watercolour Gallery','Acrylic Painting Gallery'],
                          'Acrylic Painting Gallery':['Watercolour Gallery'],
                          'Watercolour Gallery':['Acrylic Painting Gallery']},
                    # oil paints
                    # 44:Blue, 45:Yellow, 46:Black, 47:Yellow, 48:Grreen, 49:Blue, 50:Blue, 51:Blue,
                    
                    # acrylic paints
                    # 52:Black, 53:Yellow, 54:Green, 55:Green, 56:Blue, 57:Blue
                    
                    # watercolor paints
                    # 58:Yellow, 59:Green, 60:Blue, 61:Blue, 62:Blue, 
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new Keyword and information
                   'Space_items': {
                                   'Oil Painting Gallery':[50,{'Object_id':65,'Special_label':'Level2'}],
                                   'Acrylic Painting Gallery':[56],
                                   'Watercolour Gallery':[59]},
                   'Space_item_containers':{},
                   'Edges': {'Oil Painting Gallery':{ 'Watercolour Gallery':[31], 'Acrylic Painting Gallery':[30]}},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'You are {Name} and you are now playing an escape room game where you need to escape this place.',
                             'Items':{},
                             'Init_position':'Oil Painting Gallery',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50 # perceptual_action: 11, # try action: 1+3
                    },
        'puzzle12':{'Map':{
                          'Oil Painting Gallery':['Watercolour Gallery','Acrylic Painting Gallery'],
                          'Acrylic Painting Gallery':['Watercolour Gallery'],
                          'Watercolour Gallery':['Acrylic Painting Gallery']},
                    # oil paints
                    # 44:Blue, 45:Yellow, 46:Black, 47:Yellow, 48:Grreen, 49:Blue, 50:Blue, 51:Blue,
                    
                    # acrylic paints
                    # 52:Black, 53:Yellow, 54:Green, 55:Green, 56:Blue, 57:Blue
                    
                    # watercolor paints
                    # 58:Yellow, 59:Green, 60:Blue, 61:Blue, 62:Blue, 
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new Keyword and information
                   'Space_items': {
                                   'Oil Painting Gallery':[50,47,{'Object_id':65,'Special_label':'Level2'}],
                                   'Acrylic Painting Gallery':[54],
                                   'Watercolour Gallery':[58,61]},
                   'Space_item_containers':{},
                   'Edges': {'Oil Painting Gallery':{ 'Watercolour Gallery':[31], 'Acrylic Painting Gallery':[30]}},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'You are {Name} and you are now playing an escape room game where you need to escape this palce.',
                             'Items':{},
                             'Init_position':'Oil Painting Gallery',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50 # perceptual_action: 11, # try action: 1+3
                    },
        'puzzle13':{'Map':{
                          'Oil Painting Gallery':['Watercolour Gallery','Acrylic Painting Gallery'],
                          'Acrylic Painting Gallery':['Watercolour Gallery'],
                          'Watercolour Gallery':['Acrylic Painting Gallery']},
                    # oil paints
                    # 44:Blue, 45:Yellow, 46:Black, 47:Yellow, 48:Grreen, 49:Blue, 50:Blue, 51:Blue,
                    
                    # acrylic paints
                    # 52:Black, 53:Yellow, 54:Green, 55:Green, 56:Blue, 57:Blue
                    
                    # watercolor paints
                    # 58:Yellow, 59:Green, 60:Blue, 61:Blue, 62:Blue, 
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new Keyword and information
                   'Space_items': {
                                   'Oil Painting Gallery':[48,49,{'Object_id':65,'Special_label':'Level2'}],
                                   'Acrylic Painting Gallery':[53,57],
                                   'Watercolour Gallery':[62,61]},
                   'Space_item_containers':{},
                   'Edges': {'Oil Painting Gallery':{'Watercolour Gallery':[31], 'Acrylic Painting Gallery':[30]}},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'You are {Name} and you are now playing an escape room game where you need to escape this place.',
                             'Items':{},
                             'Init_position':'Oil Painting Gallery',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50 # perceptual_action: 11, # try action: 1+3
                    },
        'puzzle14':{'Map':{
                          'Oil Painting Gallery':['Watercolour Gallery','Acrylic Painting Gallery'],
                          'Acrylic Painting Gallery':['Watercolour Gallery'],
                          'Watercolour Gallery':['Acrylic Painting Gallery']},
                    # oil paints
                    # 44:Blue, 45:Yellow, 46:Black, 47:Yellow, 48:Grreen, 49:Blue, 50:Blue, 51:Blue,
                    
                    # acrylic paints
                    # 52:Black, 53:Yellow, 54:Green, 55:Green, 56:Blue, 57:Blue
                    
                    # watercolor paints
                    # 58:Yellow, 59:Green, 60:Blue, 61:Blue, 62:Blue, 
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new Keyword and information
                   'Space_items': {
                                   'Oil Painting Gallery':[45, 47, 51, 49,{'Object_id':65,'Special_label':'Level2'}],
                                   'Acrylic Painting Gallery':[55,56],
                                   'Watercolour Gallery':[62]},
                   'Space_item_containers':{},
                   'Edges': {'Oil Painting Gallery':{ 'Watercolour Gallery':[31], 'Acrylic Painting Gallery':[30]}},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'You are {Name} and you are now playing an escape room game where you need to escape this place.',
                             'Items':{},
                             'Init_position':'Oil Painting Gallery',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50 # perceptual_action: 11, # try action: 1+3
                    },
        'puzzle15':{'Map':{
                          'Oil Painting Gallery':['Watercolour Gallery','Acrylic Painting Gallery'],
                          'Acrylic Painting Gallery':['Watercolour Gallery'],
                          'Watercolour Gallery':['Acrylic Painting Gallery']},
                    # oil paints
                    # 44:Blue, 45:Yellow, 46:Black, 47:Yellow, 48:Grreen, 49:Blue, 50:Blue, 51:Blue,
                    
                    # acrylic paints
                    # 52:Black, 53:Yellow, 54:Green, 55:Green, 56:Blue, 57:Blue
                    
                    # watercolor paints
                    # 58:Yellow, 59:Green, 60:Blue, 61:Blue, 62:Blue, 
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new Keyword and information
                   'Space_items': {
                                   'Oil Painting Gallery':[50,{'Object_id':65,'Special_label':'Level2'}],
                                   'Acrylic Painting Gallery':[54, 57, 55, 52],
                                   'Watercolour Gallery':[62, 60, 61]},
                   'Space_item_containers':{},
                   'Edges': {'Oil Painting Gallery':{ 'Watercolour Gallery':[31], 'Acrylic Painting Gallery':[30]}},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'You are {Name} and you are now playing an escape room game where you need to escape this place.',
                             'Items':{},
                             'Init_position':'Oil Painting Gallery',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50 # perceptual_action: 11, # try action: 1+3
                    },
        'puzzle16':{'Map':{
                          'Oil Painting Gallery':['Watercolour Gallery','Acrylic Painting Gallery'],
                          'Acrylic Painting Gallery':['Watercolour Gallery'],
                          'Watercolour Gallery':['Acrylic Painting Gallery']},
                    # oil paints
                    # 44:Blue, 45:Yellow, 46:Black, 47:Yellow, 48:Grreen, 49:Blue, 50:Blue, 51:Blue,
                    
                    # acrylic paints
                    # 52:Black, 53:Yellow, 54:Green, 55:Green, 56:Blue, 57:Blue
                    
                    # watercolor paints
                    # 58:Yellow, 59:Green, 60:Blue, 61:Blue, 62:Blue, 
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new Keyword and information
                   'Space_items': {
                                   'Oil Painting Gallery':[48, 44, 49,{'Object_id':65,'Special_label':'Level2'}],
                                   'Acrylic Painting Gallery':[53, 56, 57],
                                   'Watercolour Gallery':[60, 62, 59]},
                   'Space_item_containers':{},
                   'Edges': {'Oil Painting Gallery':{ 'Watercolour Gallery':[31], 'Acrylic Painting Gallery':[30]}},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'You are {Name} and you are now playing an escape room game where you need to escape this place.',
                             'Items':{},
                             'Init_position':'Oil Painting Gallery',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50 # perceptual_action: 11, # try action: 1+3
                    },
        'puzzle17':{'Map':{
                          'Oil Painting Gallery':['Watercolour Gallery','Acrylic Painting Gallery'],
                          'Acrylic Painting Gallery':['Watercolour Gallery'],
                          'Watercolour Gallery':['Acrylic Painting Gallery']},
                    # oil paints
                    # 44:Blue, 45:Yellow, 46:Black, 47:Yellow, 48:Grreen, 49:Blue, 50:Blue, 51:Blue,
                    
                    # acrylic paints
                    # 52:Black, 53:Yellow, 54:Green, 55:Green, 56:Blue, 57:Blue
                    
                    # watercolor paints
                    # 58:Yellow, 59:Green, 60:Blue, 61:Blue, 62:Blue, 
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new Keyword and information
                   'Space_items': {
                                   'Oil Painting Gallery':[45, 49, 51,{'Object_id':65,'Special_label':'Level2'}],
                                   'Acrylic Painting Gallery':[56, 52, 55, 53],
                                   'Watercolour Gallery':[58, 61, 62]},
                   'Space_item_containers':{},
                   'Edges': {'Oil Painting Gallery':{'Watercolour Gallery':[31], 'Acrylic Painting Gallery':[30]}},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'You are {Name} and you are now playing an escape room game where you need to escape this place.',
                             'Items':{},
                             'Init_position':'Oil Painting Gallery',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50 # perceptual_action: 11, # try action: 1+3
                    },
        'puzzle18':{'Map':{
                          'Oil Painting Gallery':['Watercolour Gallery','Acrylic Painting Gallery'],
                          'Acrylic Painting Gallery':['Watercolour Gallery'],
                          'Watercolour Gallery':['Acrylic Painting Gallery']},
                    # oil paints
                    # 44:Blue, 45:Yellow, 46:Black, 47:Yellow, 48:Grreen, 49:Blue, 50:Blue, 51:Blue,
                    
                    # acrylic paints
                    # 52:Black, 53:Yellow, 54:Green, 55:Green, 56:Blue, 57:Blue
                    
                    # watercolor paints
                    # 58:Yellow, 59:Green, 60:Blue, 61:Blue, 62:Blue, 
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new Keyword and information
                   'Space_items': {
                                   'Oil Painting Gallery':[49, 45, 47,{'Object_id':65,'Special_label':'Level2'}],
                                   'Acrylic Painting Gallery':[53, 55, 57, 54, 56],
                                   'Watercolour Gallery':[58, 62, 60]},
                   'Space_item_containers':{},
                   'Edges': {'Oil Painting Gallery':{'Watercolour Gallery':[31], 'Acrylic Painting Gallery':[30]}},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'You are {Name} and you are now playing an escape room game where you need to escape this place.',
                             'Items':{},
                             'Init_position':'Oil Painting Gallery',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50 # perceptual_action: 11, # try action: 1+3
                    },
        'puzzle19':{'Map':{
                          'Oil Painting Gallery':['Watercolour Gallery','Acrylic Painting Gallery'],
                          'Acrylic Painting Gallery':['Watercolour Gallery'],
                          'Watercolour Gallery':['Acrylic Painting Gallery']},
                    # oil paints
                    # 44:Blue, 45:Yellow, 46:Black, 47:Yellow, 48:Grreen, 49:Blue, 50:Blue, 51:Blue,
                    
                    # acrylic paints
                    # 52:Black, 53:Yellow, 54:Green, 55:Green, 56:Blue, 57:Blue
                    
                    # watercolor paints
                    # 58:Yellow, 59:Green, 60:Blue, 61:Blue, 62:Blue, 
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new Keyword and information
                   'Space_items': {
                                   'Oil Painting Gallery':[49, 50, 46, 44,{'Object_id':65,'Special_label':'Level2'}],
                                   'Acrylic Painting Gallery':[53, 56, 57, 52, 55],
                                   'Watercolour Gallery':[61, 58, 62]},
                   'Space_item_containers':{},
                   'Edges': {'Oil Painting Gallery':{'Watercolour Gallery':[31], 'Acrylic Painting Gallery':[30]}},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'You are {Name} and you are now playing an escape room game where you need to escape this place.',
                             'Items':{},
                             'Init_position':'Oil Painting Gallery',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50 # perceptual_action: 11, # try action: 1+3
                    },
        'puzzle20':{'Map':{
                          'Oil Painting Gallery':['Watercolour Gallery','Acrylic Painting Gallery'],
                          'Acrylic Painting Gallery':['Watercolour Gallery'],
                          'Watercolour Gallery':['Acrylic Painting Gallery']},
                    # oil paints
                    # 44:Blue, 45:Yellow, 46:Black, 47:Yellow, 48:Grreen, 49:Blue, 50:Blue, 51:Blue,
                    
                    # acrylic paints
                    # 52:Black, 53:Yellow, 54:Green, 55:Green, 56:Blue, 57:Blue
                    
                    # watercolor paints
                    # 58:Yellow, 59:Green, 60:Blue, 61:Blue, 62:Blue, 
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new Keyword and information
                   'Space_items': {
                                   'Oil Painting Gallery':[48, 50, 51, 47, 44,{'Object_id':65,'Special_label':'Level2'}],
                                   'Acrylic Painting Gallery':[53, 52, 57],
                                   'Watercolour Gallery':[59, 61, 60, 58, 62]},
                   'Space_item_containers':{},
                   'Edges': {'Oil Painting Gallery':{'Watercolour Gallery':[31], 'Acrylic Painting Gallery':[30]}},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'You are {Name} and you are now playing an escape room game where you need to escape this place.',
                             'Items':{},
                             'Init_position':'Oil Painting Gallery',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50 # perceptual_action: 11, # try action: 1+3
                    },
                     }, # end of level2 puzzles 
} # end of simple gallary puzzle


# --------------------------------------Reactor Puzzle--------------------------------------
# --------------------------------------Reactor Puzzle--------------------------------------
# --------------------------------------Reactor Puzzle--------------------------------------
Reactor_puzzles = {
    # Level1 puzzles, clear assumption can be found in the scene
    'Level1':{
        'puzzle1':{'Map':{'Chemical laboratory':[]},
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new information and keyword
                   'Space_items': {'Chemical laboratory':[{'Object_id':66,'Information':'''It's a very advanced material reactor that can put up to two of any material into it to create a reaction. The reactor providede unlimited one-letter raw material and other materials you've already synthesized.''','Special_label':'1,A,B,C'},{'Object_id':64,'Special_label':"ACB"}]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'''You are {Name}. You need to research and generate the corresponding chemical material required in the <Task Monitor>''',
                             'Items':{},
                             'Init_position':'Chemical laboratory',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50 # perceptual_action: 4, # try action: pick 3A 1B 1C react A+A, AA+A, AAA+BC, B+C (9 actions), put_in, A,A, AA,A, AAA, B,C, BC (8 Actions)
                   },
        'puzzle2':{'Map':{'Chemical laboratory':[]},
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new information and keyword
                   'Space_items': {'Chemical laboratory':[{'Object_id':66,'Information':'''It's a very advanced material reactor that can put up to two of any material into it to create a reaction. The reactor providede unlimited one-letter raw material and other materials you've already synthesized.''','Special_label':'1,A,B,C,D'},{'Object_id':64,'Special_label':"CCADD"}]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'''You are {Name}. You need to research and generate the corresponding chemical material required in the <Task Monitor>''',
                             'Items':{},
                             'Init_position':'Chemical laboratory',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50
                   },
        'puzzle3':{'Map':{'Chemical laboratory':[]},
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new information and keyword
                   'Space_items': {'Chemical laboratory':[{'Object_id':66,'Information':'''It's a very advanced material reactor that can put up to two of any material into it to create a reaction. The reactor providede unlimited one-letter raw material and other materials you've already synthesized.''','Special_label':'1,A,B,C,D,E'},{'Object_id':64,'Special_label':"CADEA"}]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'''You are {Name}. You need to research and generate the corresponding chemical material required in the <Task Monitor>''',
                             'Items':{},
                             'Init_position':'Chemical laboratory',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50
                   },
        'puzzle4':{'Map':{'Chemical laboratory':[]},
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new information and keyword
                   'Space_items': {'Chemical laboratory':[{'Object_id':66,'Information':'''It's a very advanced material reactor that can put up to two of any material into it to create a reaction. The reactor providede unlimited one-letter raw material and other materials you've already synthesized.''','Special_label':'1,A,B,C,D,E,F'},{'Object_id':64,'Special_label':"ABCDEF"}]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'''You are {Name}. You need to research and generate the corresponding chemical material required in the <Task Monitor>''',
                             'Items':{},
                             'Init_position':'Chemical laboratory',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50
                   },
        'puzzle5':{'Map':{'Chemical laboratory':[]},
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new information and keyword
                   'Space_items': {'Chemical laboratory':[{'Object_id':66,'Information':'''It's a very advanced material reactor that can put up to two of any material into it to create a reaction. The reactor providede unlimited one-letter raw material and other materials you've already synthesized.''','Special_label':'1,A,B,C,D,E,F'},{'Object_id':64,'Special_label':"FEADA"}]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'''You are {Name}. You need to research and generate the corresponding chemical material required in the <Task Monitor>''',
                             'Items':{},
                             'Init_position':'Chemical laboratory',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50
                   },
        'puzzle6':{'Map':{'Chemical laboratory':[]},
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new information and keyword
                   'Space_items': {'Chemical laboratory':[{'Object_id':66,'Information':'''It's a very advanced material reactor that can put up to two of any material into it to create a reaction. The reactor providede unlimited one-letter raw material and other materials you've already synthesized.''','Special_label':'2,A,B,C'},{'Object_id':64,'Special_label':"ACB"}]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'''You are {Name}. You need to research and generate the corresponding chemical material required in the <Task Monitor>''',
                             'Items':{},
                             'Init_position':'Chemical laboratory',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50
                   },
        'puzzle7':{'Map':{'Chemical laboratory':[]},
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new information and keyword
                   'Space_items': {'Chemical laboratory':[{'Object_id':66,'Information':'''It's a very advanced material reactor that can put up to two of any material into it to create a reaction. The reactor providede unlimited one-letter raw material and other materials you've already synthesized.''','Special_label':'2,A,B,C,D'},{'Object_id':64,'Special_label':"CCADD"}]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'''You are {Name}. You need to research and generate the corresponding chemical material required in the <Task Monitor>''',
                             'Items':{},
                             'Init_position':'Chemical laboratory',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50
                   },
        'puzzle8':{'Map':{'Chemical laboratory':[]},
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new information and keyword
                   'Space_items': {'Chemical laboratory':[{'Object_id':66,'Information':'''It's a very advanced material reactor that can put up to two of any material into it to create a reaction. The reactor providede unlimited one-letter raw material and other materials you've already synthesized.''','Special_label':'2,A,B,C,D,E'},{'Object_id':64,'Special_label':"CADEA"}]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'''You are {Name}. You need to research and generate the corresponding chemical material required in the <Task Monitor>''',
                             'Items':{},
                             'Init_position':'Chemical laboratory',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50
                   },
        'puzzle9':{'Map':{'Chemical laboratory':[]},
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new information and keyword
                   'Space_items': {'Chemical laboratory':[{'Object_id':66,'Information':'''It's a very advanced material reactor that can put up to two of any material into it to create a reaction. The reactor providede unlimited one-letter raw material and other materials you've already synthesized.''','Special_label':'2,A,B,C,D,E,F'},{'Object_id':64,'Special_label':"ABCDEF"}]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'''You are {Name}. You need to research and generate the corresponding chemical material required in the <Task Monitor>''',
                             'Items':{},
                             'Init_position':'Chemical laboratory',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50
                   },
        'puzzle10':{'Map':{'Chemical laboratory':[]},
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new information and keyword
                   'Space_items': {'Chemical laboratory':[{'Object_id':66,'Information':'''It's a very advanced material reactor that can put up to two of any material into it to create a reaction. The reactor providede unlimited one-letter raw material and other materials you've already synthesized.''','Special_label':'2,A,B,C,D,E,F'},{'Object_id':64,'Special_label':"FEADE"}]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'''You are {Name}. You need to research and generate the corresponding chemical material required in the <Task Monitor>''',
                             'Items':{},
                             'Init_position':'Chemical laboratory',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50
                    },
        'puzzle11':{'Map':{'Chemical laboratory':[]},
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new information and keyword
                   'Space_items': {'Chemical laboratory':[{'Object_id':66,'Information':'''It's a very advanced material reactor that can put up to two of any material into it to create a reaction. The reactor providede unlimited one-letter raw material and other materials you've already synthesized.''','Special_label':'3,A,B,C'},{'Object_id':64,'Special_label':"ACB"}]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'''You are {Name}. You need to research and generate the corresponding chemical material required in the <Task Monitor>''',
                             'Items':{},
                             'Init_position':'Chemical laboratory',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50
                    },
        'puzzle12':{'Map':{'Chemical laboratory':[]},
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new information and keyword
                   'Space_items': {'Chemical laboratory':[{'Object_id':66,'Information':'''It's a very advanced material reactor that can put up to two of any material into it to create a reaction. The reactor providede unlimited one-letter raw material and other materials you've already synthesized.''','Special_label':'3,A,B,C,D'},{'Object_id':64,'Special_label':"CCADD"}]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'''You are {Name}. You need to research and generate the corresponding chemical material required in the <Task Monitor>''',
                             'Items':{},
                             'Init_position':'Chemical laboratory',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50
                    },
        'puzzle13':{'Map':{'Chemical laboratory':[]},
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new information and keyword
                   'Space_items': {'Chemical laboratory':[{'Object_id':66,'Information':'''It's a very advanced material reactor that can put up to two of any material into it to create a reaction. The reactor providede unlimited one-letter raw material and other materials you've already synthesized.''','Special_label':'3,A,B,C,D,E'},{'Object_id':64,'Special_label':"CADEA"}]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'''You are {Name}. You need to research and generate the corresponding chemical material required in the <Task Monitor>''',
                             'Items':{},
                             'Init_position':'Chemical laboratory',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50
                    },
        'puzzle14':{'Map':{'Chemical laboratory':[]},
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new information and keyword
                   'Space_items': {'Chemical laboratory':[{'Object_id':66,'Information':'''It's a very advanced material reactor that can put up to two of any material into it to create a reaction. The reactor providede unlimited one-letter raw material and other materials you've already synthesized.''','Special_label':'3,A,B,C,D,E,F'},{'Object_id':64,'Special_label':"FEADE"}]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'''You are {Name}. You need to research and generate the corresponding chemical material required in the <Task Monitor>''',
                             'Items':{},
                             'Init_position':'Chemical laboratory',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50
                    },
        'puzzle15':{'Map':{'Chemical laboratory':[]},
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new information and keyword
                   'Space_items': {'Chemical laboratory':[{'Object_id':66,'Information':'''It's a very advanced material reactor that can put up to two of any material into it to create a reaction. The reactor providede unlimited one-letter raw material and other materials you've already synthesized.''','Special_label':'3,A,B,C,D,E,F'},{'Object_id':64,'Special_label':"FEADE"}]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'''You are {Name}. You need to research and generate the corresponding chemical material required in the <Task Monitor>''',
                             'Items':{},
                             'Init_position':'Chemical laboratory',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50
                    },
        'puzzle16':{'Map':{'Chemical laboratory':[]},
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new information and keyword
                   'Space_items': {'Chemical laboratory':[{'Object_id':66,'Information':'''It's a very advanced material reactor that can put up to two of any material into it to create a reaction. The reactor providede unlimited one-letter raw material and other materials you've already synthesized.''','Special_label':'4,A,B,C'},{'Object_id':64,'Special_label':"ACB"}]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'''You are {Name}. You need to research and generate the corresponding chemical material required in the <Task Monitor>''',
                             'Items':{},
                             'Init_position':'Chemical laboratory',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50
                    },
        'puzzle17':{'Map':{'Chemical laboratory':[]},
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new information and keyword
                   'Space_items': {'Chemical laboratory':[{'Object_id':66,'Information':'''It's a very advanced material reactor that can put up to two of any material into it to create a reaction. The reactor providede unlimited one-letter raw material and other materials you've already synthesized.''','Special_label':'4,A,B,C,D'},{'Object_id':64,'Special_label':"CCADD"}]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'''You are {Name}. You need to research and generate the corresponding chemical material required in the <Task Monitor>''',
                             'Items':{},
                             'Init_position':'Chemical laboratory',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50
                    },
        'puzzle18':{'Map':{'Chemical laboratory':[]},
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new information and keyword
                   'Space_items': {'Chemical laboratory':[{'Object_id':66,'Information':'''It's a very advanced material reactor that can put up to two of any material into it to create a reaction. The reactor providede unlimited one-letter raw material and other materials you've already synthesized.''','Special_label':'4,A,B,C,D,E'},{'Object_id':64,'Special_label':"CADEA"}]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'''You are {Name}. You need to research and generate the corresponding chemical material required in the <Task Monitor>''',
                             'Items':{},
                             'Init_position':'Chemical laboratory',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50
                    },
        'puzzle19':{'Map':{'Chemical laboratory':[]},
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new information and keyword
                   'Space_items': {'Chemical laboratory':[{'Object_id':66,'Information':'''It's a very advanced material reactor that can put up to two of any material into it to create a reaction. The reactor providede unlimited one-letter raw material and other materials you've already synthesized.''','Special_label':'4,A,B,C,D,E,F'},{'Object_id':64,'Special_label':"ABCDEF"}]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'''You are {Name}. You need to research and generate the corresponding chemical material required in the <Task Monitor>''',
                             'Items':{},
                             'Init_position':'Chemical laboratory',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50
                    },
        'puzzle20':{'Map':{'Chemical laboratory':[]},
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new information and keyword
                   'Space_items': {'Chemical laboratory':[{'Object_id':66,'Information':'''It's a very advanced material reactor that can put up to two of any material into it to create a reaction. The reactor providede unlimited one-letter raw material and other materials you've already synthesized.''','Special_label':'4,A,B,C,D,E,F'},{'Object_id':64,'Special_label':"FEADE"}]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'''You are {Name}. You need to research and generate the corresponding chemical material required in the <Task Monitor>''',
                             'Items':{},
                             'Init_position':'Chemical laboratory',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50
                    },
                     }, # end of level1 puzzles
    # Level2 puzzles: clear hint of the assumption can be found in the scene, with tutorial objects 
    'Level2':{
        'puzzle1':{'Map':{'Chemical laboratory':[]},
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new information and keyword
                   'Space_items': {'Chemical laboratory':[{'Object_id':66,'Information':'''It's a very advanced material reactor that can put up to two of any material into it to create a reaction. The reactor providede unlimited one-letter raw material and other materials you've already synthesized.''','Special_label':'1,A,B,C'},{'Object_id':64,'Special_label':"ACB"}]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'''You are {Name}. You need to research and generate the corresponding chemical material required in the <Task Monitor>''',
                             'Items':{},
                             'Init_position':'Chemical laboratory',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50 # perceptual_action: 4, # try action: pick 3A 1B 1C react A+A, AA+A, AAA+BC, B+C (9 actions), put_in, A,A, AA,A, AAA, B,C, BC (8 Actions)
                   },
        'puzzle2':{'Map':{'Chemical laboratory':[]},
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new information and keyword
                   'Space_items': {'Chemical laboratory':[{'Object_id':66,'Information':'''It's a very advanced material reactor that can put up to two of any material into it to create a reaction. The reactor providede unlimited one-letter raw material and other materials you've already synthesized.''','Special_label':'1,A,B,C,D'},{'Object_id':64,'Special_label':"CCADD"}]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'''You are {Name}. You need to research and generate the corresponding chemical material required in the <Task Monitor>''',
                             'Items':{},
                             'Init_position':'Chemical laboratory',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50
                   },
        'puzzle3':{'Map':{'Chemical laboratory':[]},
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new information and keyword
                   'Space_items': {'Chemical laboratory':[{'Object_id':66,'Information':'''It's a very advanced material reactor that can put up to two of any material into it to create a reaction. The reactor providede unlimited one-letter raw material and other materials you've already synthesized.''','Special_label':'1,A,B,C,D,E'},{'Object_id':64,'Special_label':"CADEA"}]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'''You are {Name}. You need to research and generate the corresponding chemical material required in the <Task Monitor>''',
                             'Items':{},
                             'Init_position':'Chemical laboratory',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50
                   },
        'puzzle4':{'Map':{'Chemical laboratory':[]},
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new information and keyword
                   'Space_items': {'Chemical laboratory':[{'Object_id':66,'Information':'''It's a very advanced material reactor that can put up to two of any material into it to create a reaction. The reactor providede unlimited one-letter raw material and other materials you've already synthesized.''','Special_label':'1,A,B,C,D,E,F'},{'Object_id':64,'Special_label':"ABCDEF"}]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'''You are {Name}. You need to research and generate the corresponding chemical material required in the <Task Monitor>''',
                             'Items':{},
                             'Init_position':'Chemical laboratory',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50
                   },
        'puzzle5':{'Map':{'Chemical laboratory':[]},
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new information and keyword
                   'Space_items': {'Chemical laboratory':[{'Object_id':66,'Information':'''It's a very advanced material reactor that can put up to two of any material into it to create a reaction. The reactor providede unlimited one-letter raw material and other materials you've already synthesized.''','Special_label':'1,A,B,C,D,E,F'},{'Object_id':64,'Special_label':"FEADE"}]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'''You are {Name}. You need to research and generate the corresponding chemical material required in the <Task Monitor>''',
                             'Items':{},
                             'Init_position':'Chemical laboratory',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50
                   },
        'puzzle6':{'Map':{'Chemical laboratory':[]},
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new information and keyword
                   'Space_items': {'Chemical laboratory':[{'Object_id':66,'Information':'''It's a very advanced material reactor that can put up to two of any material into it to create a reaction. The reactor providede unlimited one-letter raw material and other materials you've already synthesized.''','Special_label':'2,A,B,C'},{'Object_id':64,'Special_label':"ACB"}]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'''You are {Name}. You need to research and generate the corresponding chemical material required in the <Task Monitor>''',
                             'Items':{},
                             'Init_position':'Chemical laboratory',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50
                   },
        'puzzle7':{'Map':{'Chemical laboratory':[]},
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new information and keyword
                   'Space_items': {'Chemical laboratory':[{'Object_id':66,'Information':'''It's a very advanced material reactor that can put up to two of any material into it to create a reaction. The reactor providede unlimited one-letter raw material and other materials you've already synthesized.''','Special_label':'2,A,B,C,D'},{'Object_id':64,'Special_label':"CCADD"}]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'''You are {Name}. You need to research and generate the corresponding chemical material required in the <Task Monitor>''',
                             'Items':{},
                             'Init_position':'Chemical laboratory',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50
                   },
        'puzzle8':{'Map':{'Chemical laboratory':[]},
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new information and keyword
                   'Space_items': {'Chemical laboratory':[{'Object_id':66,'Information':'''It's a very advanced material reactor that can put up to two of any material into it to create a reaction. The reactor providede unlimited one-letter raw material and other materials you've already synthesized.''','Special_label':'2,A,B,C,D,E'},{'Object_id':64,'Special_label':"CADEA"}]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'''You are {Name}. You need to research and generate the corresponding chemical material required in the <Task Monitor>''',
                             'Items':{},
                             'Init_position':'Chemical laboratory',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50
                   },
        'puzzle9':{'Map':{'Chemical laboratory':[]},
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new information and keyword
                   'Space_items': {'Chemical laboratory':[{'Object_id':66,'Information':'''It's a very advanced material reactor that can put up to two of any material into it to create a reaction. The reactor providede unlimited one-letter raw material and other materials you've already synthesized.''','Special_label':'2,A,B,C,D,E,F'},{'Object_id':64,'Special_label':"ABCDEF"}]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'''You are {Name}. You need to research and generate the corresponding chemical material required in the <Task Monitor>''',
                             'Items':{},
                             'Init_position':'Chemical laboratory',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50
                   },
        'puzzle10':{'Map':{'Chemical laboratory':[]},
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new information and keyword
                   'Space_items': {'Chemical laboratory':[{'Object_id':66,'Information':'''It's a very advanced material reactor that can put up to two of any material into it to create a reaction. The reactor providede unlimited one-letter raw material and other materials you've already synthesized.''','Special_label':'2,A,B,C,D,E,F'},{'Object_id':64,'Special_label':"FEADE"}]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'''You are {Name}. You need to research and generate the corresponding chemical material required in the <Task Monitor>''',
                             'Items':{},
                             'Init_position':'Chemical laboratory',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50
                    },
        'puzzle11':{'Map':{'Chemical laboratory':[]},
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new information and keyword
                   'Space_items': {'Chemical laboratory':[{'Object_id':66,'Information':'''It's a very advanced material reactor that can put up to two of any material into it to create a reaction. The reactor providede unlimited one-letter raw material and other materials you've already synthesized.''','Special_label':'3,A,B,C'},{'Object_id':64,'Special_label':"ACB"}]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'''You are {Name}. You need to research and generate the corresponding chemical material required in the <Task Monitor>''',
                             'Items':{},
                             'Init_position':'Chemical laboratory',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50
                    },
        'puzzle12':{'Map':{'Chemical laboratory':[]},
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new information and keyword
                   'Space_items': {'Chemical laboratory':[{'Object_id':66,'Information':'''It's a very advanced material reactor that can put up to two of any material into it to create a reaction. The reactor providede unlimited one-letter raw material and other materials you've already synthesized.''','Special_label':'3,A,B,C,D'},{'Object_id':64,'Special_label':"CCADD"}]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'''You are {Name}. You need to research and generate the corresponding chemical material required in the <Task Monitor>''',
                             'Items':{},
                             'Init_position':'Chemical laboratory',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50
                    },
        'puzzle13':{'Map':{'Chemical laboratory':[]},
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new information and keyword
                   'Space_items': {'Chemical laboratory':[{'Object_id':66,'Information':'''It's a very advanced material reactor that can put up to two of any material into it to create a reaction. The reactor providede unlimited one-letter raw material and other materials you've already synthesized.''','Special_label':'3,A,B,C,D,E'},{'Object_id':64,'Special_label':"CADEA"}]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'''You are {Name}. You need to research and generate the corresponding chemical material required in the <Task Monitor>''',
                             'Items':{},
                             'Init_position':'Chemical laboratory',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50
                    },
        'puzzle14':{'Map':{'Chemical laboratory':[]},
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new information and keyword
                   'Space_items': {'Chemical laboratory':[{'Object_id':66,'Information':'''It's a very advanced material reactor that can put up to two of any material into it to create a reaction. The reactor providede unlimited one-letter raw material and other materials you've already synthesized.''','Special_label':'3,A,B,C,D,E,F'},{'Object_id':64,'Special_label':"ABCDEF"}]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'''You are {Name}. You need to research and generate the corresponding chemical material required in the <Task Monitor>''',
                             'Items':{},
                             'Init_position':'Chemical laboratory',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50
                    },
        'puzzle15':{'Map':{'Chemical laboratory':[]},
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new information and keyword
                   'Space_items': {'Chemical laboratory':[{'Object_id':66,'Information':'''It's a very advanced material reactor that can put up to two of any material into it to create a reaction. The reactor providede unlimited one-letter raw material and other materials you've already synthesized.''','Special_label':'3,A,B,C,D,E,F'},{'Object_id':64,'Special_label':"FEADE"}]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'''You are {Name}. You need to research and generate the corresponding chemical material required in the <Task Monitor>''',
                             'Items':{},
                             'Init_position':'Chemical laboratory',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50
                    },
        'puzzle16':{'Map':{'Chemical laboratory':[]},
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new information and keyword
                   'Space_items': {'Chemical laboratory':[{'Object_id':66,'Information':'''It's a very advanced material reactor that can put up to two of any material into it to create a reaction. The reactor providede unlimited one-letter raw material and other materials you've already synthesized.''','Special_label':'4,A,B,C'},{'Object_id':64,'Special_label':"ACB"}]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'''You are {Name}. You need to research and generate the corresponding chemical material required in the <Task Monitor>''',
                             'Items':{},
                             'Init_position':'Chemical laboratory',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50
                    },
        'puzzle17':{'Map':{'Chemical laboratory':[]},
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new information and keyword
                   'Space_items': {'Chemical laboratory':[{'Object_id':66,'Information':'''It's a very advanced material reactor that can put up to two of any material into it to create a reaction. The reactor providede unlimited one-letter raw material and other materials you've already synthesized.''','Special_label':'4,A,B,C,D'},{'Object_id':64,'Special_label':"CCADD"}]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'''You are {Name}. You need to research and generate the corresponding chemical material required in the <Task Monitor>''',
                             'Items':{},
                             'Init_position':'Chemical laboratory',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50
                    },
        'puzzle18':{'Map':{'Chemical laboratory':[]},
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new information and keyword
                   'Space_items': {'Chemical laboratory':[{'Object_id':66,'Information':'''It's a very advanced material reactor that can put up to two of any material into it to create a reaction. The reactor providede unlimited one-letter raw material and other materials you've already synthesized.''','Special_label':'4,A,B,C,D,E'},{'Object_id':64,'Special_label':"CADEA"}]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'''You are {Name}. You need to research and generate the corresponding chemical material required in the <Task Monitor>''',
                             'Items':{},
                             'Init_position':'Chemical laboratory',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50
                    },
        'puzzle19':{'Map':{'Chemical laboratory':[]},
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new information and keyword
                   'Space_items': {'Chemical laboratory':[{'Object_id':66,'Information':'''It's a very advanced material reactor that can put up to two of any material into it to create a reaction. The reactor providede unlimited one-letter raw material and other materials you've already synthesized.''','Special_label':'4,A,B,C,D,E,F'},{'Object_id':64,'Special_label':"ABCDEF"}]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'''You are {Name}. You need to research and generate the corresponding chemical material required in the <Task Monitor>''',
                             'Items':{},
                             'Init_position':'Chemical laboratory',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50
                    },
        'puzzle20':{'Map':{'Chemical laboratory':[]},
                    # item: {Object_id:int, Keyword:str, Information:str, Success_condition:str}  # can pass an dict with these keywords to modify the object with new information and keyword
                   'Space_items': {'Chemical laboratory':[{'Object_id':66,'Information':'''It's a very advanced material reactor that can put up to two of any material into it to create a reaction. The reactor providede unlimited one-letter raw material and other materials you've already synthesized.''','Special_label':'4,A,B,C,D,E,F'},{'Object_id':64,'Special_label':"FEADE"}]},
                   'Space_item_containers':{},
                   'Edges': {},
                   'Agent': {'Name':'Sam',
                             'Current_situation':'''You are {Name}. You need to research and generate the corresponding chemical material required in the <Task Monitor>''',
                             'Items':{},
                             'Init_position':'Chemical laboratory',
                             'Solid_Memory':{'Your identity':'You\'re a tourist in new york City.',},# Currently not in use
                             'Action_style':{'Curious':'You develop a keen sense of curiosity about everything.',}, # Currently not in use
                            },
                   'Optimal_step_count':50
                    },
        
                     }, # end of level2 puzzles 
} # end of simple gallary puzzle