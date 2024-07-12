+++
title = "Software Mistakes and Tradeoffs"
description = "Good software is all about making tradeoffs.  This book helps you make better choices and reviews common software mistakes. "
+++

# I.  THE BIG PICTURE
### Chapter 1: What Would You Say You Do Here?
- Staff : 1st line manager, Senior Staff : Senior Manager; Principle : Director
- Do we even need a staff engineer position?
    -  a job title vests authority in people who might not automatically receive
    -  saves them the time and energy they would otherwise have to spend proving themselves again and again
- Why do we staff need to see big picture?
    - To find a ==global maximum==, which is the
        - good for the org and not 1 team
        - over time
    - Good decisions need _context_. Experienced engineers know that the answer to most technology choices is “it depends.” Knowing the pros and cons of a particular technology isn’t enough—you need to know the local details too. What are you trying to do? How much ==time, money, and patience== do you have? What’s your ==risk tolerance==? What does the business need?
- What's the staff engineer's role across projects?
    - Use their experience to ==anticipate risks and ask hard questions==
    - When the project gets stuck, they have enough perspective to ==track down the causes and unblock==
    - Evengalize: telling the story of what’s happening and why, selling the vision to the rest of the company, and explaining what the work will make possible and how the new project affects everyone
- Why the staff engineer needs to be good at influencing
    - Within engineer circle, Staff engineers are ==role models==. Managers may be responsible for setting culture on their teams, enforcing good behavior, and ensuring standards are met. But engineering norms are set by the behavior of the most respected engineers on the project.
- What is your Role?
    - Staff can report to VP, Directors, Senior Manager, Manager
    - Which level one is on influences
        - what problem you are asked of
        - what information you are privy to
        - how much time you may get from your manager
- What is Your Scope
    - Your reporting chain will likely affect your _scope_: the domain, team, or teams that you pay close attention to and have some responsibility for
    - Pros and Cons of scopes
        - scope too narrow -> lack of impoact, overshadowing other engineers
        - scope too broad -> lack of impact, becoming a bottleneck, decision fatigue
- What's Your Primary Focus?
    - What's important? our work needs to be important” doesn’t mean you should only work on the fanciest, most glamorous technologies and VP-sponsored initiatives. The work that’s most important will often be the work that nobody else see
        - Be wary of choosing a project that already has a lot of senior people on it


### Chapter 2: Three Maps
- Map
    - Where we came from
    - Where we started working together
    - Where are are
    - Where we're going

- Recap
    - Practice the skills of intentionally looking for a bigger picture and seeing what’s happening.
    - Understand your work in context: know your customers, talk with peers outside your group, understand your success metrics, and be clear on what’s actually important.
    - Know how your organization works and how decisions get made within it.
    - Build or discover paths to allow information you need to come to you.
    - Be clear about what goals everyone is aiming for.
    - Think about your own work and what your journey is.



### Chapter 3: Creating the Big Picture
- This chapter addresses when the goal is not clear OR if plan is disputed?
- A Common Reincarnation of the Same Problem
    - All teams share the same platform (ie Salesforce Hyperforce)
        - Con
            - Hyperforce deployment can become a copy and paste exercise;
            - Code repo is a single source of error; so needs more verification/validationshuman approvals.
            - can suffer from generalization penalty,
    - Decentralized, every org use their own proprietary (ie Ebay core Search)
        - cons:
            -  There can be a [“tragedy of the commons”](https://oreil.ly/KEPbi), where teams pursuing their own best action without coordinating with others leads to an outcome that is bad for everyone. ==There’s that local maximum again==.
            -  Shared concerns can be neglected because no one group has the authority or an incentive to fix them alone.
            - Teams may be missing enough context to make the best decision. The people taking action might be different from the people experiencing the outcomes, or might be separated in time from them.
- Technical vision vs Strategy
    - Technical vision describes the future one strives once the objectives have been achieved and biggest problem solved
        - Examples of better future states:
            - better state for your architecture, code, processes, technology, and teams
        - With so many staff and architecits, it is crucial to write down and be detailed about
    - Strategy is plan of action. It is the method and metrics in V2MOM.
        - Strategies needs to be realistic and acknowledge the constraints of your situation.
        - Aspects of a strategy (from _Good Strategy/Bad Strategy_, Rumelt)
            - the diagnosis
                - “What’s going on here?” The diagnosis of your situation needs to be simpler than the messy reality, perhaps by finding patterns in the noise or using metaphors or mental models to make the problem easy to understand. You’re trying to distill the situation you’re in down to its most essential characteristics so that it’s possible to really comprehend it. This is _difficult_.
            - guiding policy / design philosopy
                - The guiding policy is your approach to bypassing the obstacles described in the diagnosis. It should give you a clear direction and make the decisions that follow easier. Rumelt says it should be short and clear, “a signpost, marking the direction forward.”
            - coherent actions you are going to take
- Prep work
    - Embrace the boring/existing ideas
        - ```I kind of think writing about engineering strategy is hard because good strategy is pretty boring, and it’s kind of boring to write about. Also I think when people hear “strategy” they think “innovation.” If you write something interesting, it’s probably wrong. This seems directionally correct, certainly most strategy is synthesis and that point isn’t often made! ```
    - Join an Expedition in Progress
    - Get a Sponsor
        -  Maximize your chances by bringing the potential sponsor something they want'
        - Find out what’s blocking their goals, and see if the problem you’re trying to solve lines up with those. The sponsor will also have some “objectives that are always true”: if you can make their teams more productive or (genuinely) happier
    - Choose Your Core Group
        - Keep each accountable
        - Working in a group also lets you pool your contacts, social capital, and credibility, so you can reach a broader swathe of the organization
        - Who do you need on your side? Who’s going to be opposed? If there are dissenters who will hate any decision made without them, you have two options: make very sure that you have enough support to counter their naysaying, or bring them along from the start.
        - Be clear from the start about whether you consider yourself the lead and ultimate decider of this project or more of a [first among equals](https://oreil.ly/BKNe3) If you’re later going to want to use tiebreaker powers or pull rank, highlight your role as lead from the start, perhaps by adding a “roles” section to whatever kickoff documentation you’re creating. But whether you’re the lead or not, let your group have ideas, drive the project forward, and talk about it without redirecting all questions to you
    - Choose Your Secondary Group
        - As for your broader group of allies, keep them engaged: interview them and use the information they give you. Send them updates on how the document is going. Invite them to comment on early drafts. Your path will be much easier if you have a group of influential people who support what you’re doing, and the knowledge they bring from across the organization will yield better results, too
    - Make sure it's achievable


- Writing Cycle
   ```mermaid
      graph TD;
          1_Initial_Ideas --> 2_Write;
          4_Think_Process_Synthesize_Decide --> 2_Write;
          2_Write --> 3_TalkToPeople_Interview_Align;
          2_Write --> 5_Share_Broadly;
          3_TalkToPeople_Interview_Align --> 4_Think_Process_Synthesize_Decide;
  ```
    - Initial Ideas
        - What documents already exist?
        - What needs to change?  What's difficult now?  Knowing what your company does and where the team should invest.
        - What's Important?
        - What will Future You wish Present You had done?
    - Writing
    - Interviews
        - Early questions:
            - We’re creating a plan for X. What’s important to include?
            - What’s it like working with…?”
            - If you could wave a magic wand, what would be different about…?
    - Arriving to Decisions
        - Tradeoffs
            - decisions are hard because no matter what you choose, there will be disadvantages.   Priorities help decide which disadvantage you're willing to accept, and vice versa.
                - use even or statement
                - We will optmize X even over Y
        - Building Consensus
            - ==determining==  consensus and ==coming to== consensus are different things than   ==having== consensus [IETF](https://datatracker.ietf.org/doc/html/rfc7282#page-4)  The key is to separate those  choices that are simply unappealing from those that are truly  problematic
            - Application: Rather than asking, “Is everyone OK with choice A?” ask, “Can anyone not live with choice A?”
            - Sometimes it is more important moving forward than one person's arguent carrying the day.
        - Getting and Staying Aligned
            - Understand who your final audience is?  local team;, product maanger
            - Keep sponser up to date and what you're planning and how it's going
                - Take the time to get your thoughts together so that you can bring them a summary of how you’re approaching various problems and what your options ar
            - Be aspirational but not impossible
            - [Nemawashi](https://en.wikipedia.org/wiki/Nemawashi) : means sharing information and layting the foudnations so that by the time a decision is made, there's already a consensus of opinion.
            - Alignment goes both directions
        - Make sure the story is comprehensible, relatable, and comfortable.


# II. EXECUTION
### Chapter 4: Finite Time
- You can't do it all. As a senior person, you can see a host of problems: architectures that won’t scale, processes that waste everyone’s time, missed opportunities
- If you commit to something, what are you implicitly saying no to?
- Your 5 resources
    - Energy
    - Credibility
        - You can build credibility by solving hard problems, being visibly competent, and consistently showing good technical judgment.
        - One cause of loss of credibility is absolutism. Ex: always advocating one technology in all situations.
        -  If you’re polite (even to annoying people), communicate clearly, and stay calm in stressful situations, other people will trust you to be the adult in the room.
        - At staff+ levels, you’re often trying to find a balance between seeing the big picture and accepting pragmatic local solutions.
    - Quality of Life
        - money can improve your quality of life by giving you control and acheive other priorities.
    - Skills
        - Our industry moves fast. Any technical skill set slowly becomes less relevant and eventually gets out of date.
        - Three ways to improve skills
            1. Structured: class, buy a book, hack on a project
                1. con: eats into your free time
            2. Working with someone who is really skilled
            3. (most common)  Learn by doing;
                1. ie on a project, sprint tickets
    - Social Capital
        - Vs Credibility
            - While _credibility_ is whether others think you’re capable of doing whatever you’re trying to do, _social capital_ reflects whether they want to help you do it
        - Social capital is a mix of trust, friendship, and that feeling of owing someone a favor, or of believing they’ll remember that they owe you one.
        - Some ways to build capital
            - spend time with people, have good conversations, help them out, support each other
            - Completing projects;
                - ```"nothing breeds succeeds likes success"```
- ==When you take on a project, you need to be aware of how the project add/subtract to the 5 resources: (skills, quality of life, social capitol, credibility, energy)==.
- If you’re leading the project, you’ll be filling in the gap.  (Remember Ebay)
- By the time you reach the staff+ level, you will be largely (but probably not entirely) responsible for choosing your own work. How do you choose?  Look at the 5 resources. Examples and nuances are in book
    - Is this fight worth wit
    - Sponsorhip; goes both ways as a sponsor and and sponsered: you need to deliver to get more of it.
- Recap
    - By the time you reach the staff+ level, you will be largely (but probably not entirely) responsible for choosing your own work. This includes deciding on the extent and duration of your involvement with any given project. You can’t do everything, so you’ll need to choose your battles.
    - You are responsible for choosing work that aligns with your life and career needs as well as advancing your company’s goals.
    - You are responsible for managing your energy.
    - Some projects will make you happier than others, or improve your quality of life more.
    - Your social capital and peer credibility are “bank accounts” that you can pay into and spend from. You can lend credibility and social capital to other people, but be conscious of whom you’re lending to.
    - Skills come from taking on projects and from deliberately learning. Make sure you’re building the skills you want to have.
    - Free up your resources by giving other people opportunities to grow, including starting projects and handing them off.
    - Focus means sometimes saying no. Learn how.


### Chapter 5: Leading Big Projects
#### 5.1 Life of a Project

#### 5.2 At the Start of a Project
- Building Context
    - Goals: why is this the one that is happening and not others
        - Customer needs:
        - Success metrics: describe sucess is measured?
        - Constraints:  deadlines, resources, other responsibilities, other team's dependencies
        - Risks: Try to predict some of the risks
            - What could happen that could prevent you from reaching your goals on deadline?
            - Are there unknowns, dependencies, key people who will doom the project if they quit? You can mitigate risk by being clear about your areas of uncertainty.
            - What don’t you know, and what approaches can you take that will make those less ambiguous?
            - Is this the sort of thing you can prototype? Has someone done this kind of project before
        - History:
            - where did the idea come from?
            - Have we tried it before?
            - If I am new to existing project, respect what came before
                - findout what system you've can use
                - understand people's feelings and expectations, and learn from their experiences

- Project Structure
    - Defining Roles on your Team Via Responsibility Assignment Matrix (RACI)
        - Responsible: The person actually doing the work.
        - Accountable: The person ultimately delivering the work and responsible for signing off that it’s done. There’s supposed to be only one accountable person per task, and it will often be the same person as whoever is “Responsible.”
        - Consulted: People who are asked for their opinion.
        - Informed: People who will be kept up to date on progress.

    - Recruiting people/leads
        - When you’re responsible for a really big project, you’re kind of building an organization and you’re steering an organization. It’s important to get the needs of the organization right. I spent a lot of time trying to make sure that I had the best leads in all the different domains that were involved in that one specific project. And, when you’re looking for leads, you’re looking for people that ==don’t just have good technical judgment==, but also ==have the attitudes: they are optimistic, good at conflict resolution, good at communication==

    - Estimating Time
        - Pragmatic Programmer:
            - We find that often the only way to determine the timetable for a project is by gaining experience on that same project
        - Take into consideration the team you depend on; how is their plate?  their history?
    - Logistics of Team
        - When, where, and how you'll meet
        - Channels of informal communication
        - How you share status
        - Where the documentation home will be
        - What is the development practice like?


#### 5.3 Driving the Project
- Exploring
    - When project just start off,  you won’t have enough information about it on day one to make  granular decisions (ie use technologyX to build featureY)
    - What are some important aspects of the project?
        - Bigger projects with multiple teams, where each team may have a different mental model.  Get to the point where you can ==concisely== explain what different teams in the project want in a way that they’ll agree is accurate.
        - Alighning and framing takes time and effort.  talk to users and stakeholders. listen to what they say and how they say it.
        - ==Develop a crisp definition of what you're doing.==
            - Understanding the problem well helps you frame it for other people
    - What possible approaches can you take?
        - Be open to existing solutions, even if less convenient or interesting

- Clarifying
    - As lead, it is your responsibility to create a mental model to what you are doing
        - How can you hook this project into their current knowledge
    - Mental models:
        - Connecting an abstract idea back to something I understand well removes some of the cognitive cost of retaining and describing the idea
        - Describe interfaces without the messy details
        - Give people a head start by providing a convenient, memorable name for the concept, using an analogy, or connecting it back to something they already understand
    - Naming
        - Two people can use the same words and mean quite different things
    - Picutre and graphs
        - reduce complexity with pictures
- Designing
    - Design doc s the most efficient way to make sure everyone is aligned.
        - ven if you’re not hearing objections when you talk about them, your colleagues may not have internalized the plan and their implicit agreement may not mean anything
    - Be precise
        - It is better to be wrong than be vague
        - Tips:
            1. Use active voice; Be clear ==who/what== is doing the action
                1. before: the jason payload will be unpacked
                2. better: the parse component will upack the JSON payload
            2. Use a few extra words or even repeat yourself if it means avoiding ambiguity
                1. Ex: add a word after ambiguous words like this or that
                    1. To solve this ==shortage==, we should order more.
        - Writing References
            - [[Tool - Technical Writing]]
    - Security/Privacy/Compliance
        - What data are being exposed?
    - Common Design Pitalls
        - Building from scratch. Most problems are not brand new; learn from other people and consider reuse existing problems
        - Lack of understanding can make a hard task appear trivial
        - Building for the present without consideration for the future
        - Building for the distant distant future
        - Ignoring the difficult part
        - Being non-realistic of the workload
        - Adopting an approach/system that is not really designed for it.
        - Create something that is hard to maintain and understand (ie mc-kernel)
        - bikeshedding; spening the most resource on the most trivial, easiest reversible decisions.

- Coding (as a staff engineer)
    - Staff engineer sets the example
        - Meet or exceed the testing standards
        - Add useful comments and documentation
    - Should you code?
        - if it excites you and you are not the bottleneck
        - as a cartographer; to gain deeper knowledge and develop better road maps and action items.
        - To develop a pattern, so other people can implement
        - Be careful not to take the most important and biggest work; you could be dragged into other issues and you want to give people room to grow

- Communicating
    - Communicating well is key for delivering a project in time. If the teams aren’t talking to each other, you won’t make progress. And if you’re not talking to people outside the project, your stakeholders won’t know how it’s going
    - Talking to each other (especially for remote)
        - Talking to each other builds familarity, which helps people safer to ask clarifying questions.
        - To help people talk
            - friendly slack channels
            - demo
            - social events
    - Sharing status to stakeholders, sponsers, and customers
        - Explain status in terms of impact
        - On information the audience actually want to know

- Navigating (when something goes wrong)
    - Tech leads are responsible for rerouting, escalating to someone who can help, or, if you really have to, breaking the news to your stakeholders that the goal is now unreachable
    - The bigger the change, the more likely it is that you’ll need to go back to the start of this chapter, deal with the overwhelm all over again, build context, and treat the project like it’s restarting.
        - Don’t create panic or let rumors start. Instead, give people facts and be clear about what you want them to do with the information.
    -


### Chapter 6: Why Have We Stopped
- Recap
    -  As the project lead, you are responsible for understanding why your project has stopped and getting it started again.
    -  As a leader in your organization, you can help restart other people’s projects too.
    - You can unblock projects by
        - explaining what needs to happen
        - reducing what other people need to do
        - clarifying the organizational support, escalating
        - making alternative plans.
    - You can bring clarity to a project that’s adrift by defining your destination, agreeing on roles, and asking for help when you need it.
    - Don’t declare victory unless the project is truly complete. Code completeness is just one milestone.
    - Whether you’re ready or not, sometimes it’s time for the project to end. Celebrate, retrospect, and clean up.


# III. LEVELING UP
- I believe title refers to leveling the team up
    - Note: to level up team, you also need to level yourself up because the staff engineer sets the standard

### Chapter 7: You are a Role Model
#### 7.1 Be Competent (Technically)
- Know Things
    - Your big-picture thinking, project execution, credibility, and influence are underpinned by knowledge and experience
    - Build Experience
        - Technical skill comes from study and practice
        - Experience comes through time, exposure (projects), and study
            - Be aware of too early of promotion to staff, as it may rob you of hands on building experience
        - Being competent and knowledgeable doesn’t mean you have to know the most about every topic.  Can you learn the technology, domain etc..
    - Build domain knowledge
        - Leverage your experience/technical foundation to come up to speed in the domain
        - Learn the technology in whatever way is most effective for you.
        - Get a sense of the appropriate trade-offs; the resource constraints; the common arguments, biases, and in-jokes.
        - Know which technologies are on the market and how they might be used. Read the core literature.
        - Know who the “famous people” in the domain are and what they advocate for. Twitter is good for this
    - Stay up to date
        - It’s embarrassing for everyone when a technical leader insists on a “best practice” that has been debunked for a decade or a technology that everyone else has moved on from
        - Even if you aren’t deep in the code any more, your spidey-sense should stay sharp for “code smells” or problems waiting to happen
        - be wary of drifting so far from the technology that you’re only learning how _your company_ operates
- Be Self Aware
    - Admit what you know:
        - be ==confident== and ==honest== with yourself with what you know and what you don't
    - Admit what you don't know
        - be ok with asking, explain it like I'm 5 year old
    - Understand your own context
        - You have a perspective, that your context is not the universal context, and that your opinions and knowledge are specific to _you_.

- Have High Standards
    - Your standard will serve as a model for how other people work
    - Seek out constructive criticism
        - ask for code reviews, design review, and peer evaluation
    - Own your mistake
        - If you react well and fix the problem you caused, you could even end up with _more_ esteem from your colleague
    - Be reilable
        - Finish what you start

#### 7.2 Be Responsible
- Take Ownership
    - Senior people own the whole problem, not just the parts that go as planned. You’re not running someone else’s project for them: it’s _yours_ and you don’t passively let it sink or swim
    - Avoid "Cover Your Ass"
        - Mature engineers stand up and accept the responsibility given to them. If they find they don’t have the requisite authority to be held accountable for their work, they seek out ways to rectify that. An example of CYAE is “It’s not my fault. They broke it, they used it wrong. I built it to spec, I can’t be held responsible for their mistakes or improper specification.”
    - Use you own judgement; don't need to constantly ask for permission
    - Ask obvious questions; as senior, you can ask questions that are so obvious, no one else is willing to ask
    - Do the glue work; tasks that are needed to make team successful
        - a lot of the work that’s not anybody’s job but that furthers your goals. Redirect your junior colleagues toward tasks that will develop their careers instead.


- Take Charge
    - When you se a gap, step in to fill the gap
    - Step up in an emergency
    - Drive meetings
        - with a clear agenda
    - When something disrespectful or offensive in a public channel, one should speak up

- Create Calm
    - defuse, don't amplify
        - when you're dealing with a big problem, try to make it smaller
        - be cautious where/how you share your anxieties/frustration.
            - acknowledge there are problems
            - be clear about the intention of the sharing
                - want someone to take action
                - bounce ideas
                - sharing context
    - Be consistent and predictable
        - since it's hard to be consistent when stresed, make sure to take time off, get enough rest.

#### 7.3 Remember the Goal
- There's the Business
    - adapt to the situation
        - phases: grow, acquisition, new markets
    - has a budget
    - so spend mindfully

* There's the User
    * Know who uses your software and how they use it.
    * Make sure the feature you are creating is actually needed

* There's the Team
    * You are working as a team; don't be single point of failure
    * be aware of the capability of your team

#### 7.4 Look Ahead
- Anticipate what the Future You wish Present You had done
    - Be clear what the your broad direction is, even if you don't know the details
    - Take time to optimize your workflow
        - codebase
            - style guide
        - documentation
    - Keep your tools sharp
        - make your environment better measured by velocity and reliability
    - Create institutional memory
- Expect Failure
    - Plan in advance for major incidents by adding some conventions around how you work together during an emergency: introduce the incident command system I mentioned earlier, for example, and practice the response before you need it

- Optimize for Maintenance, Not Creation
    - Make it understandable
        1. classes and hands on experiences
        2. make it as easy as possible for people to understand the system when they need to
            1. documentation(s)
                1. short intro
                2. a big deep one
    -  Make system observable
        1. easy to inspect, analyze, and debug
    - Make system simple
        - Good programmers write code that humans can understand.”[12](https://learning.oreilly.com/library/view/the-staff-engineers/9781098118723/ch07.html#ch01fn98) Senior engineers sometimes think they can demonstrate their prowess with the flashiest, most complicated solutions.
        - Make things simple by spending more time on it; working solution is just the first iteration

#### 7.5 Recap
- Your words and actions carry more weight now. Be deliberate.
- Invest the time to build knowledge and expertise. Competence comes from experience.
- Be self-aware about what you know and what you don’t.
- Strive to be consistent, reliable, and trustworthy.
- Get comfortable taking charge when nobody else is, including during a crisis or an ambiguous project.
- When someone needs to say something, say something.
- Create calm. Make problems smaller, not bigger.
- Be aware of your business, budgets, user needs, and the capabilities of your


### Chapter 8: Good Influence At Scale
#### 8.1 Good Influence

#### 8.2 Advice


#### 8.3 Teaching


#### 8.4 Guardrails


#### 8.5 Opportunity

#### 8.6 Recap
-   You can help your colleagues by providing advice, teaching, guardrails, or opportunities. Understand what’s most helpful for the specific situation.

-   Think about whether you want to help one-on-one, level up your team, or influence further.

-   Offer your experience and advice, but make sure it’s welcome. Writing and public speaking can send your message further.

-   Teach through pairing, shadowing, review, and coaching. Teaching classes or writing codelabs can scale your teaching time.

-   Guardrails can let people work autonomously. Offer review, or be a project guardrail for your colleagues. Codify guardrails using processes, automation, and culture change.

-   Opportunities can be much more valuable than advice. Think about who you’re sponsoring and delegating to. Share the spotlight in your team.

-   Plan to give away your job.