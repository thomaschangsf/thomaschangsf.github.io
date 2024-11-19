+++
title = "Making Better Prediction"
description = "Some Thoughts on How to Make Better Predictions"
+++

# Background
Humans, much like machine learning models, are constantly making predictions. These range from simpler guesses about the weather to complex decisions, like managing financial investments by weighing risks and rewards. Improving our ability to predict can improve the quality of our life. This blog explores how we can enhance our predictions by applying machine learning practices to both our internal mindsets and external situations.

# Internal
### Mindset
Our approach to making predictions can vary depending on the nature of the problem. Sometimes, adopting a "hedgehog" perspective—focusing on one strong, confident idea—works well. This approach keeps things simple and straightforward, which can be effective in contexts like social media, where clarity and confidence are valued.

Other times, we benefit from a "fox" perspective, which involves seeing things from multiple angles. This approach allows us to consider various perspectives and blend conflicting ideas into a cohesive view. Fox-like thinkers tend to use multiple tools, evaluate probabilities, and recognize the assumptions behind their predictions.

Let us see how the hedgehog and fox analogy look from the lense of the statistical ensemble learning models. 

In a random forest model, multiple decision trees work together to improve prediction accuracy. Each tree is built using a different subset of data and focuses on a unique set of "signals" or features within the data. By selecting different signals, individual trees may focus on various aspects of the data, capturing diverse patterns or nuances that a single model might overlook. When combined, this collection of trees brings multiple perspectives to the prediction process, much like the "fox" approach that values looking at things from many angles. The random forest leverages diversity among its trees to generate more robust predictions, reducing the risk of overfitting and making it adaptable to a range of situations.

In contrast, a gradient-boosted tree model operates more like the "hedgehog" perspective, refining predictions iteratively in a single, focused direction. Each tree in a gradient-boosted model is trained to correct the errors of the previous one, systematically honing in on improving a specific aspect of the model's performance. Rather than looking at diverse signals, gradient boosting focuses on progressively reducing residual errors, with each tree becoming more narrowly targeted to achieve the best possible prediction. This method exemplifies the "hedgehog" mindset, where a single, dedicated approach is applied in depth, honing in on a particular goal for increasingly precise outcomes. 


## Track Your Predictions
To improve prediction accuracy, it’s important to keep track of your forecasts. Make specific predictions, including measurable details like dates, so you can assess their accuracy over time. Quantifying predictions can involve metrics like the Brier Score, which measures how well-calibrated your predictions are. For example, if you state something with 80% confidence, check whether it turns out to be true 80% of the time. Beware of relying only on averages, as they can sometimes mask important details in your predictions.

Closing the model performance gap in the real world with the offline training environment is crucial for any ML project. The model performs well in the offline training environment, achieving high accuracy and low error rates on the training dataset. However, when deployed in the real world, the model’s performance may drop significantly. Two example in the ecommerce field is the seasonality or user preference changes effect. Potential mitigations includes real time feature offline online integrations, more regular model training/deployment, bootstrap sampling strategies etc..    

## Growth Mindset and Grit
Effective predictors also have a growth mindset. They believe in their ability to improve and actively seek ways to refine their skills. Alongside this, they possess grit—the determination to keep going, even when predictions don’t always pan out. This combination of resilience and a willingness to learn helps in continuously making better, more informed predictions.

Machine learning models, like deep neural networks, improve with more data. But just as models can get stuck in local maxima, we too can hit plateaus, feeling stuck in our routines. I've found that curiosity-driven exploration—whether in modeling, services, pipelines, infrastructure, or products—can be a powerful way to overcome inertia and unlock new growth.

# External
## Cooperation Without Deference
In team settings, it's essential to cooperate without simply deferring to others. Striving for consensus can sometimes hinder good decision-making, as it may lead to settling on ideas without fully exploring alternatives. Disagreement, on the other hand, can be productive and even necessary. Teams benefit from differing opinions as long as the debate remains respectful.

Teams should aim to disagree without becoming disagreeable. This means focusing on data and past experiences rather than personal biases. Constructive debate strengthens predictions by allowing teams to examine our assumptions and understandings more deeply.

## Diversity Drives Better Predictions
Diverse teams bring a range of perspectives, especially when tackling complex problems that span multiple areas of expertise. Each member can contribute unique insights, which together help build a more complete picture. The more varied the viewpoints within a team, the greater the chances of uncovering critical pieces of information that improve the accuracy of forecasts.

## Leading Teams
### Balancing Leadership and Forecasting
Effective leadership often seems at odds with good forecasting. Leaders are expected to project confidence and set an unifying vision, often in situations when the picture is incomplete and changes are dynamic.

Good forecasters see situations as complex and uncertain, often thinking in terms of probabilities. This awareness can make decisions harder, as they consider many factors that might affect outcomes. They approach decisions with humility and a broad perspective, sometimes called 'dragonfly eyes,' seeing from multiple angles. However, this can also lead to over-analysis or relying too much on group decisions

### Balancing Humility and Confidence
The key to harmonizing leadership and forecasting skills lies in knowing when to be confident and when to be humble. Intellectual humility is important in fields like machine learning, where complexity and the number of variables involved require a growth mindset. Adopting a forecaster's perspective means being open to continuous learning and staying adaptable.

At the same time, it’s important to have confidence. Confidence, in my opinion, should be grounded through preparation and past succeses. For instance, Steph Curry has often credited his confidence to his consistent work ethic and preparation. He believes that confidence comes from the hours he puts into practicing his skills and refining his game. Curry has spoken about his commitment to perfecting his shooting and maintaining a disciplined training routine, which he says gives him a sense of assurance on the court. TThis type of confidence gives me the foundation to compete with anyone. If I don’t succeed, I can learn from the experience.


### Leading a Team with Auftragstaktik
I have observed throughout my careers how a company delegate the decision-making process to have a strong correlation to their ability to execute.  These companies tend to follow a variant of the military inspired Auftragstaktik approach

In this approach, the high-level strategy and goals are set by leaders at the top. Leaders at this level outline the direction, discuss options, and set the standard. Execution decisions, however, are delegated to those closer to the action. This approach relies on the assumption that team members on the ground have relevant information and critical thinking skills, allowing for faster and more effective responses. The effectiveness of this method, however, depends on the talent, motivation, etc.. of the team members.  I think this is one reason why some companies relies on processes to mitigate the talent gap. 

Leadership at both the strategy and execution levels involves two main phases. In the strategy phase, leaders focus on forecasting, weighing possibilities and outcomes to set a strong foundation. The execution phase, in contrast, is simpler but demands determination and relentless action. Once a strategy is in place, team members execute it thoroughly until they achieve data or specific milestones. At this point, it’s essential to review progress, analyze data, and apply different levels of thinking to assess the situation.

The ability to adapt and pivot based on new insights is crucial, as demonstrated by companies like Amazon. This flexibility allows teams to blend conflicting ideas, refine their approach, and stay aligned with the evolving demands of their mission.