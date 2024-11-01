+++
title = "Thinking About Thinking"
description = "Some Thoughts on Mastery"
+++

# Background
ML knowledge and capabilities have grown exponentially with unprecedented rate of increase in model abilities, research output, compute resources, and real-world applications. ML related research papers have increased at an annual rate of 35% since 2012.  Model parameters, as a first order estimate for model complexities and capabilities, has grown roughly 10 times every 1-2 years.  Models like GPT-4 and Claude has achieved near human scores on challenging tasks, such as standardized exams and domain specific benchmarks like GLUE (General Language Understanding Evaluation) and MMLU (Massive Multitask Language Understanding).
The rate of growth will likely continue to grow as the data model application flywheel is further optimized and companies continue to invest tremendous resources in AI initiatives.

In today’s rapidly evolving knowledge landscape, I’m interested in understanding which types of human insights can better enhance LLM capabilities. How do we currently assess subject expertise, and how might that change in the future? I will explore these questions over time in this blog. By articulating and applying these ideas daily, I strive to develop expertise in a subject more efficiently and effectively.

## Knowledge Taxonomy
Bloom's taxonomy offers a structured approach to assess and develop expertise by organizing knowledge into levels, ranging from fundamental understanding to complex analysis and creation. Let's explore how this framework applies in the machine learning context.

### Level 1: Memorization
Memorization relies on repeating information until it sticks, using tools like flashcards or re-reading material. While basic facts—like syntax, functions, and key terms—help form a foundation, they’re not enough for real-world problem-solving. Today, with easy access to information through resources like ChatGPT, memorization holds less value than it once did, especially in machine learning where application and deeper understanding matter most.

I believe this type of knowledge is essential for building a foundation in any domain, but its value has gradually declined as search engines and LLMs now make information easily accessible.

### Level 2: Comprehension
With comprehension, we start to understand information, going beyond facts to grasp what they mean. In machine learning, comprehension shows up when we can explain essential concepts—like overfitting, underfitting, or gradient descent—in simple terms. Testing comprehension by explaining ideas to someone else, even a child, reinforces our understanding. This level aligns with how we typically learn in college, where we’re encouraged to make sense of complex concepts rather than just remember them. 

True understanding, to me, means being able to break down complex ideas into simpler components that anyone, especially non-ML practitioners, can grasp. When tackling complex problems, it’s essential to comprehend the big picture while also identifying the key concerns to optimize given the constraints.

### Level 3: Application
Application involves using what we know to solve problems. In machine learning, this ranges from solving basic coding problems to tackling complex projects. Applying knowledge might mean implementing a linear regression model, experimenting with different data-cleaning techniques, or practicing coding challenges that require a mix of data structures and algorithms. Each problem we tackle builds a bridge between our theoretical knowledge and real-world skills.  

Simple applications tend to require one ML concept or domain.  Solving complex ML problems often, in my experience, requires one to draw on multiple ML domains: data, cloud native applications, modeling, infrastructure, and user understanding. An example of this is the LLM. LLM combines several key ML technologies: transformer architecture for natural language abilities, reinforcement learning for alignment, supervised/unsupervised learning for instruction tuning, and large scale distributed training and inferencing optimizations.  

Let's assume we have fine tune an image classification model with transformers. One way to boost performance is to increase the size of the model. This often requires one to scale training across multiple GPUs or machines, making the most of available processing resources. Another way to improve the model is by using more training data, which may require for efficient data loading and parallel processing, commonly done with Spark pipelines. However, these pipelines can slow down iteration speeds due to feature iteration loops and the scala/java/python/framework development tax. To deploy the model at scale, it’s crucial to ensure it runs reliably and uses resources efficiently. We may also need to address differences between online and offline environments, like feature and codebase variations. Finally, deployed models must be monitored for issues, which require investigating the data, model, app, or infrastructure stack.


### Level 4: Analysis
Analysis requires us to compare, contrast, and identify relationships. In machine learning, this might look like evaluating different algorithms, model architectures, or data-preprocessing methods. Tools like Venn diagrams, summaries, and comparison tables help visualize these relationships and deepen our understanding. We have to think critically about each component and its role within a set of constraints. Which algorithm families should we explore for this domain problem?  What component should we do offline or online dependencies?

### Level 5: Evaluation
Evaluation asks us to think critically and prioritize. We start asking questions like “Why does this approach work best?” or “How can we improve it?” By prioritizing, we build decision-making skills, which become essential for real-world applications. For example, in machine learning, we might evaluate and justify a choice between model interpretability and accuracy, deciding which holds more value for the specific problem. Creating mind maps and questioning assumptions can help us refine these evaluative skills.

In my experience across various projects, effective prioritization often depends on the project stage. For greenfield projects, it's crucial to focus on rapid prototyping, gathering stakeholder/user feedbacks, validating hypotheses, and extracting quick insights. For mature projects, priorities shift to defining optimization metrics that closely reflect business goals and building ML code infrastructure that maximize scalability, reliability, maintainability, and high iteration speed.

### Level 6: Creation
The highest level, creation, means generating something new—whether it’s a hypothesis, a novel solution, or an innovative model. In machine learning, this could involve designing new algorithms, hypothesizing methods to handle unusual data, or even identifying gaps in existing solutions. At this stage, we become active contributors, moving from consumers of knowledge to creators and innovators. Examples of this are paper submission to technical conferences or patents.

# Conclusion
Bloom's taxonomy offers an effective method for assessing the value of knowledge. By cultivating intentions and skills at level 3 and above, we can enhance our capabilities in learning, problem-solving, and decision-making.