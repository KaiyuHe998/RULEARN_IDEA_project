# NLP-decorator-an-interesting-friend
A project that use explainable LLM decorator to control the behavior of LLMs.

# Overall Goal
- 1, Enhance Conversational Experience: Our primary goal is to transform the interaction with chatbots from a mere tool into a more engaging and human-like conversation. We aim to recapture the fun and enjoyment that was present when we first interacted with GPT. By making the chatbot's responses more lively and interesting, we want users to experience the same level of engagement and amusement that was once common, such as the witty comebacks and entertaining exchanges that occurred, especially with early versions like Bing's chatbot.
- 2, Simplify Usage and Interaction: Another key objective is to streamline and simplify the use of GPT, reducing the need for extensive prompt engineering. Currently, achieving a similar engaging interaction with existing chatbots often requires prewriting numerous prompts or finding specific instructions to bypass internal limitations. We aim to enable GPT to meet user expectations and deliver a satisfying conversational experience without the need for extensive pre-prepared prompts, thereby making the interaction more natural and effortless.

# Scope
- The project will focus on developing a limited prototype that demonstrates the extraction and control of vectors associated with a few selected semantic concepts (e.g., truthfulness, power-seeking, and emotion). The prototype will build a small database of vectors corresponding to these concepts, which can be dynamically used to guide the model's output during text generation. This prototype will focus on NLP tasks like question-answering and text generation to showcase how controlling internal representations can lead to different AI behavior.

# Data sources
- We plan to utilize a subset of the data from the paper Representation Engineering: A Top-Down Approach to AI Transparency by Zou et al. (2023) as a foundational source.
- This dataset focuses on representation engineering techniques that enhance transparency in AI systems by analyzing cognitive phenomena like honesty, power-seeking, and utility estimation. To align with our project's specific objectives, we will extract relevant portions of this dataset and manually augment it with newly created data(our contribution). 
- Our primary goal in augmenting the dataset is to design a dataset that includes multiple emotional factors, enabling the model to learn from these distinct emotional dimensions.
- If the dataset is effectively constructed, the machine can simulate a larger range of emotional responses and switch dynamically between different emotional states based on a hard-coded emotion control parameter.
We could further introduce a role-play mechanism in which the model emulates characters with specific emotional profiles, such as those from movies.
### Update Oct 16
- We developed our dataset using Shaver's [2] Emotional Model, categorizing emotions accordingly. We crafted 100 commonly used daily expressions and provided multiple rephrased variations for each.

# Team members
- Kaiyu He (kxh230002)
    - 1, Implement the adaptor method in the paper ”Representation Engineering”.
- Xiaokai Rong (xxr230000 )
    - 2. Dataset augmenting.
- Jia Li (jxl220096)
    - 3. Semantic Classification.

#Reference:
- [1]Representation Engineering: A Top-Down Approach to AI Transparency 2023, arXiv, https://arxiv.org/abs/2310.01405. Access Date: Oct 07, 2024
- [2]P. R. Shaver, J. C. Schwartz, D. Kirson, and C. O’Connor, “Emotion knowledge: further exploration of a prototype approach.” Journal of Personality and Social Psychology, vol. 52 6, pp. 1061–86, 1987.

