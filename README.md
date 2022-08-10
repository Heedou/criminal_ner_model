# Criminal NER Model
This chapter describes how to develop an entity name recognition system in the criminal domain.
In this process, a new training data set was invented, and an NER experiment using pretrained language models was conducted.

# 1. Entity Definition
In terms of crime analysis, new entities can be defined.
Therefore, crime-related information that can be extracted from Korean crime facts is suggested. The result is shown in the figure below.

![image](https://user-images.githubusercontent.com/49702343/183605409-41553f9f-ec12-45ca-9b3e-dbe3f50c272f.png)

# 2. Data Collection & Building Training Data

## Data Collection
Criminal facts texts were collected from criminal sentences. By using sentence-split python library, crime facts paragraphs separated into sentences.

## Building Training Data
In this process, a NLP Tagging Tool was made using python-pyqt5 library.
More information about the tagging tool can be found at the link below.

First, the rule-base approach was used initially.
Second, it was corrected by criminal analysts 

![image](https://user-images.githubusercontent.com/49702343/183605603-2f2d0121-0837-4ec9-a4af-ff5bd9dc466d.png)
