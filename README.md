# Fairness in Human Activity Recognition

In this tutorial, we will build a method for quantifying the fairness of Human Activity Recognition models with respect to gender through the use of a gender classifier. We wil begin with a brief background on HAR systems and fairness (and why it's important to measure fairness in HAR) followed by a high-level overview of the system that we will be building. Once we have a general understanding of the system, we will jump right into building it. 

## Background
### Why Fairness?
To begin, let's first discuss the need for measuring fairness in the context of computer vision. If you do a quick google images search for "bias in computer vision", you'll see numerous examples of how bias seeps into CV systems when depoloyed into the real world (some examples below). When constructing CV systems, it's imperative that these systems perform consistently across all groups of people (e.g. race, gender, age) to prevent biased outcomes. 

### Understanding HAR
The central goal behind **Human Activity/Action Recognition** is pretty intuitive - we aim to teach computers to recognize human activities such as cooking, biking, jumping jacks, etc. The applications of this technology are plentiful: crime detection, recognizing critical health conditions, and motion-based control are just a few. 

Most HAR systems used today contain three main components: A **Data Source** (Input), an **Activity Classifier** (Model), and a **Class Prediction** (Output), where the Activity Classifier is the key component in predicting the activity. 

In the following section, we will extend this system such that it enables us to measure the fairness of the Activity Classifier with respect to gender. 

## System Architecture
In order to extend this system to measure HAR fairness, we will first define the three components 