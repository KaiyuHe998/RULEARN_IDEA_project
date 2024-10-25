# RULEARN-IDEA
This is the official repository for "[IDEA: Enhancing the Rule Learning Ability of Large Language Model Agent through Induction, Deduction, and Abduction]([https://arxiv.org/abs/2310.01405](https://arxiv.org/abs/2408.10455))"  
by Kaiyu He, Mian Zhang, Shuo Yan, Peilin Wu, and Zhiyu Chen.

<img align="center" src="figures/RULEARN_IDEA_introduction.png" width="750">

## Introduction
We introduce RULEARN, a novel benchmark specifically designed to assess the rule-learning abilities of LLM agents in interactive settings. In RULEARN, agents strategically interact with simulated environments to gather observations, discern patterns, and solve complex problems. To enhance the rule-learning capabilities for LLM agents, we propose IDEA, a novel reasoning framework that integrates the process of **I**nduction, **DE**duction, and **A**bduction. The IDEA agent generates initial hypotheses from limited observations through abduction, devises plans to validate these hypotheses or leverages them to solve problems via deduction, and refines previous hypotheses using patterns identified from new observations through induction, dynamically establishing and applying rules that mimic human rule-learning behaviors. Our evaluation of the IDEA framework, which involves five representative LLMs, demonstrates significant improvements over the baseline.

## Installation

```bash
git clone https://github.com/MeanStudent/RULEARN_IDEA_project.git
cd RULEARN_IDEA_project
pip install -r requirements.txt.
```

## Quickstart

An introductory example is provided in example_notebook.ipynb. Feel free to run a single puzzle with selected models to explore the RULEARN environment in action.

You can also run the following code to solve the puzzle yourself. This is the same code we used with human participants in our study.

```bash
python RULEARN_IDEA/human_test.py
```

## Reproduction

The code runs a batch of puzzles in parrllele. If you want to reproduce the result, please specify the model name and your openai token and otherthings needed in the RULEARN_IDEA/run_experiments.py
You can using the following code to run the experiment
```bash
python RULEARN_IDEA/run_experiments.py
```

## Create your own object and puzzles

Some predefined objects are listed in `data/CHIBI_database.xlsx`. If you simply want to add a new object with a unique name or description, you can add a new line to the database file. For objects that only require a basic message flow (e.g., taking a certain action results in a specific message), you can create them by modifying only the `data/CHIBI_database.xlsx` file.

However, for more complex objects (like the reactor used in our study) that require system-level variable changes, you'll need to create a new subclass that inherits from the Fixed_Interact_Pipeline_Object_Base class in `fixed_interactive_pipeline_objects.py`. This setup allows you to work with system-level variables to build a state machine object, and you can also add custom variables for more complex effects.
For example, here’s a state machine object (a animal) that automatically finds and eats an object with name you specified in the `data/CHIBI_database.xlsx` (for example we have a sheep will automatically eat cabbage when no agent seen in the same space defined in the database):

```python
class StateMachineObjectAnimal(StateMachineObjectsBase):
    """
    A state machine object that mimics animal behavior; when no agent is in the same space,
    it will seek out and consume a target object. You can add more template actions and 
    interactive actions for agents, such as patting or riding the animal.
    """

    # CHIBI Object Interfaces ---------------------------------------------------
    def show(self):
        print(self.Keyword)
        print(self.Information)
        print(self.Parse_pipeline_dict)
        print(self.State_machine_actions)

    def destroy(self):
        pass  # Additional inherited functions as needed

    def systemic_parse_state_machine(self, state_machine_action_information: Dict[str, Any]):
        """
        This function enables effects when specified conditions in the database are met.
        For example, if the 'Systemic_parse_id' is 'eat', the animal will locate and
        consume the target object.
        """
        if state_machine_action_information['Systemic_parse_id'] == 'eat':
            linked_objects = self.find_linked_object_state_machine(state_machine_action_information)
            object_to_be_eaten = list(linked_objects.values())[0]  # Only one target object for the "eat" action
            object_to_be_eaten.destroy()
            raise utils.TaskCompletedException(f'Mission failed: {self.get_keyword()} ate {object_to_be_eaten.get_keyword()}')

    def systemic_parse(self, ...):
        """
        Defines additional interactive actions for agents, inheriting from the base class.
        For example, this animal can be patted or ridden by an agent.
        """
        pass

```
To set up a new interactive environment, follow the dictionary structure in `all_puzzle_settings.py`. Define the space names, edges, and populate it with objects you’ve defined.
## Citation
If you find this useful in your research, please consider citing:

```
@misc{he2024ideaenhancingrulelearning,
      title={IDEA: Enhancing the Rule Learning Ability of Large Language Model Agent through Induction, Deduction, and Abduction}, 
      author={Kaiyu He and Mian Zhang and Shuo Yan and Peilin Wu and Zhiyu Zoey Chen},
      year={2024},
      eprint={2408.10455},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2408.10455}, 
}
```

