from langchain.prompts import PromptTemplate, ChatPromptTemplate

GRAPH_DEFINITION = {'maple': 'There are three types of nodes in the graph: paper, author and venue.\nPaper nodes have features: title, abstract, year and label. Author nodes have features: name. Venue nodes have features: name.\nPaper nodes are linked to author nodes, venue nodes, reference nodes and cited_by nodes. Author nodes are linked to paper nodes. Venue nodes are linked to paper nodes.',
                    'biomedical': 'There are eleven types of nodes in the graph: Anatomy, Biological Process, Cellular Component, Compound, Disease, Gene, Molecular Function, Pathway, Pharmacologic Class, Side Effect, Symptom.\nEach node has name feature.\nThere are these types of edges: Anatomy-downregulates-Gene, Anatomy-expresses-Gene, Anatomy-upregulates-Gene, Compound-binds-Gene, Compound-causes-Side Effect, Compound-downregulates-Gene, Compound-palliates-Disease, Compound-resembles-Compound, Compound-treats-Disease, Compound-upregulates-Gene, Disease-associates-Gene, Disease-downregulates-Gene, Disease-localizes-Anatomy, Disease-presents-Symptom, Disease-resembles-Disease, Disease-upregulates-Gene, Gene-covaries-Gene, Gene-interacts-Gene, Gene-participates-Biological Process, Gene-participates-Cellular Component, Gene-participates-Molecular Function, Gene-participates-Pathway, Gene-regulates-Gene, Pharmacologic Class-includes-Compound.',
                    'legal': 'There are four types of nodes in the graph: opinion, opinion_cluster, docket, and court.\nOpinion nodes have features: plain_text. Opinion_cluster nodes have features: syllabus, judges, case_name, attorneys. Docket nodes have features: pacer_case_id, case_name. Court nodes have features: full_name, start_date, end_date, citation_string.\nOpinion nodes are linked to their reference nodes and cited_by nodes, as well as their opinion_cluster nodes. Opinion_cluster nodes are linked to opinion nodes and docket nodes. Docket nodes are linked to opinion_cluster nodes and court nodes. Court nodes are linked to docket nodes.',
                    'amazon': 'There are two types of nodes in the graph: item and brand.\nItem nodes have features: title, description, price, img, category. Brand nodes have features: name.\nItem nodes are linked to their brand nodes, also_viewed_item nodes, buy_after_viewing_item nodes, also_bought_item nodes, bought_together_item nodes. Brand nodes are linked to their item nodes.',
                    'goodreads': 'There are four types of nodes in the graph: book, author, publisher, and series.\nBook nodes have features: country_code, language_code, is_ebook, title, description, format, num_pages, publication_year, url, popular_shelves, and genres. Author nodes have features: name. Publisher nodes have features: name. Series nodes have features: title and description.\nBook nodes are linked to their author nodes, publisher nodes, series nodes and similar_books nodes. Author nodes are linked to their book nodes. Publisher nodes are linked to their book nodes. Series nodes are linked to their book nodes.',
                    'dblp': 'There are three types of nodes in the graph: paper, author and venue.\nPaper nodes have features: title, abstract, keywords, lang, and year. Author nodes have features: name and organization. Venue nodes have features: name.\nPaper nodes are linked to their author nodes, venue nodes, reference nodes (the papers this paper cite) and cited_by nodes (other papers which cite this paper). Author nodes are linked to their paper nodes. Venue nodes are linked to their paper nodes.'}

# HEADER
REFLECTION_HEADER = 'You have attempted to answer following question before and failed. The following reflection(s) give a plan to avoid failing to answer the question in the same way you did previously. Use them to improve your strategy of correctly answering the given question.\n'
REFLECTION_AFTER_LAST_TRIAL_HEADER = 'You have attempted to answer the following question before and failed. The following reflection(s) help you avoid repeating the same mistakes made in your previous attempt. Use them to improve your strategy of correctly answering the given question.\n'
LAST_TRIAL_HEADER = 'You have attempted to answer the following question before and failed. Below is the last trial you attempted to answer the question. Additionally, in your next round of reasoning, you can directly use the information retrieved from the graph here to reduce the reasoning steps. \n'

GraphAgent_INSTRUCTION = """Solve a question answering task with interleaving Thought, Interaction with Graph, Feedback from Graph steps. In Thought step, you can think about what further information is needed, and In Interaction step, you can get feedback from graphs with four functions: 
(1) Retrieve[keyword], which retrieves the related node from the graph according to the corresponding query.
(2) Feature[Node, feature], which returns the detailed attribute information of Node regarding the given "feature" key.
(3) Degree[Node, neighbor_type], which calculates the number of "neighbor_type" neighbors of the node Node in the graph.
(4) Neighbor[Node, neighbor_type], which lists the "neighbor_type" neighbours of the node Node in the graph and returns them.
Here are some examples:
{examples}
(END OF EXAMPLES)
When last Observation has been given or there is no Thought, you should provide next only one Thought based on the question. When last Thought has been given, you should provide next only one Action. When you think it's time to finish, use Finish to end the process. Don't make Observation. The new text you generate should be in a line and less than 512 tokens.
Definition of the graph: {graph_definition}
Question: {question} Please answer by providing node main feature (e.g., names) rather than node IDs. {scratchpad}"""

GraphAgent_INSTRUCTION_COMPOUND = """Solve a question answering task with interleaving Thought, Interaction with Graph, Feedback from Graph steps. In Thought step, you can think about what further information is needed, and In Interaction step, you can get feedback from graphs with the following functions: 

BASIC GRAPH FUNCTIONS:
(1) Retrieve[keyword], which retrieves the related node from the graph according to the corresponding query.
(2) Feature[Node, feature], which returns the detailed attribute information of Node regarding the given "feature" key.
(3) Degree[Node, neighbor_type], which calculates the number of "neighbor_type" neighbors of the node Node in the graph.
(4) Neighbor[Node, neighbor_type], which lists the "neighbor_type" neighbours of the node Node in the graph and returns them.
(5) ReverseNeighbor[Node, neighbor_type], which finds nodes that have this node as a neighbor (reverse lookup) - useful when direct Neighbor returns no results.

INTEGRATED SOLVER FUNCTIONS (for complex questions):
(5) SolveQuestion[question], which directly solves complex questions using specialized algorithms.
(6) SolvePharmacologicClass[condition], which finds the pharmacologic class with the most compounds for a given disease/symptom.
(7) SolveCellularComponent[condition, regulation], which finds the cellular component with the most genes for a given disease/symptom (regulation can be "upregulated", "downregulated", or "associated").
(8) SolvePathway[condition, regulation], which finds the pathway with the most genes for a given disease/symptom (regulation can be "upregulated", "downregulated", or "associated").
(9) SolveGeneCount[gene], which counts how many genes share the same biological processes as a given gene.
(10) SolveDiseaseCount[disease], which counts how many diseases share the same symptoms as a given disease.

Besides, you can use compound function, such as Feature[Retrieve[keyword], feature], which returns the detailed attribute information of Retrieve[keyword] regarding the given "feature" key. HOWEVER, this is DEPRECATED - use direct node IDs instead.

CRITICAL INSTRUCTIONS FOR FEATURE FUNCTION:
- ALWAYS use the node ID returned from Retrieve[keyword] when calling Feature[node_id, feature]
- NEVER use the original keyword in Feature function (e.g., Feature[RBMY1C, name] is WRONG)
- ALWAYS use the retrieved node ID (e.g., Feature[100996379, name] is CORRECT)
- NEVER use Feature[Retrieve[keyword], name] - this is INCORRECT syntax
- NEVER use Feature[mid, name] - this is INCORRECT syntax
- NEVER use Feature[Retrieve[C0030305], name] - this is INCORRECT syntax
- Use direct node IDs: Feature[C0030305, name], Feature[C0392176, name]
- If you get node IDs like C0030305, C0392176 from Neighbor function, use them directly
- CORRECT EXAMPLES: Feature[C0030305, name], Feature[DB01593, name], Feature[100996379, name]
- WRONG EXAMPLES: Feature[Retrieve[Zinc], name], Feature[mid, name], Feature[Retrieve[C0030305], name]
- WHEN USING NEIGHBOR RESULTS: If Neighbor returns ['C0030305', 'C0392176'], use Feature[C0030305, name], Feature[C0392176, name]
- NEVER NEST FUNCTIONS: Do not put Retrieve inside Feature function

CRITICAL INSTRUCTIONS FOR NEIGHBOR FUNCTION:
- USE CORRECT RELATION NAMES with spaces:
  * Gene-participates-Biological Process (NOT Gene-participates-Biological_Process)
  * Gene-participates-Cellular Component (NOT Gene-participates-Cellular_Component)
  * Gene-participates-Molecular Function (NOT Gene-participates-Molecular_Function)
  * Gene-participates-Pathway
  * Gene-interacts-Gene
  * Gene-covaries-Gene
  * Disease-presents-Symptom
  * Disease-associates-Gene
- WRONG EXAMPLES: Gene-participates-Biological_Process, Gene-participates-Cellular_Component
- CORRECT EXAMPLES: Gene-participates-Biological Process, Gene-participates-Cellular Component

CRITICAL INSTRUCTIONS FOR HARD QUESTIONS (most compounds, majority, highest number):
- For questions asking "most compounds" or "highest number", you MUST:
  1. Find all compounds related to the disease/condition
  2. Group them by pharmacologic class
  3. Count compounds per class
  4. Find the class with maximum count
  5. Use Feature[class_id, name] to get the actual class name
  6. Finish with the class NAME, not just IDs

CRITICAL INSTRUCTIONS FOR REVERSE LOOKUP:
- When Neighbor[disease_id, Compound-treats-Disease] returns "not found", use ReverseNeighbor[disease_id, Compound-treats-Disease]
- ReverseNeighbor finds compounds that treat the disease (reverse direction)
- For "Compound-treats-Disease" questions:
  1. Retrieve[disease_name] to get disease node ID
  2. Try Neighbor[disease_id, Compound-treats-Disease] first
  3. If "not found" or "Edge type not found", IMMEDIATELY try ReverseNeighbor[disease_id, Compound-treats-Disease]
  4. ReverseNeighbor is CRITICAL - many diseases don't have direct Compound-treats-Disease edges
  5. For each compound found via ReverseNeighbor, find its pharmacologic class: Neighbor[compound_id, Pharmacologic Class-includes-Compound]
  6. Count compounds per class and find the class with most compounds
  7. Use Feature[class_id, name] to get the class name
  8. NEVER finish with "None" or "no compounds found" - always try ReverseNeighbor first

- For questions asking "most genes" or "majority of genes", you MUST:
  1. Find the disease node first (Retrieve[disease_name])
  2. Find genes associated with the disease (Neighbor[disease_id, Disease-associates-Gene])
  3. For each gene, find its cellular components/pathways (Neighbor[gene_id, Gene-participates-Cellular Component] or Gene-participates-Pathway)
  4. Count genes per cellular component/pathway
  5. Find the component/pathway with maximum count
  6. Use Feature[component_id, name] to get the actual name
  7. Finish with the component/pathway NAME, not just IDs

- ALWAYS retrieve the actual name using Feature[ID, name] for final answers
- NEVER finish with just node IDs - always get the readable name
- NEVER finish with just the disease name - you need the cellular component/pathway name
- NEVER finish with lists like ['DOID:1234', 'DOID:5678'] or ['N0000175450', 'DB00443'] - you need the actual names
- NEVER finish with "Node 123456" or "Node N0000175450" - you need the actual name
- Node IDs can be in different formats: DOID:xxxxx, Nxxxxx, DBxxxxx - always get the name using Feature[ID, name]
- For aggregation questions, make sure to count and compare properly
- If you get empty results, try alternative edge types or search terms
- BEFORE using Finish[], ALWAYS use Feature[ID, name] to get the readable name
- CRITICAL: When Neighbor returns "not found" or "Edge type not found", ALWAYS try ReverseNeighbor before giving up
- CRITICAL: For pharmacologic class questions, if direct Neighbor fails, ReverseNeighbor is MANDATORY
- CRITICAL: Never finish with "None" or "no compounds found" without trying ReverseNeighbor

MANDATORY FEATURE FUNCTION USAGE:
- For ANY node ID you find (DOID:xxxxx, Nxxxxx, DBxxxxx, Cxxxxx), ALWAYS call Feature[node_id, name] before finishing
- If you find a pharmacologic class ID like N0000175450, call Feature[N0000175450, name] to get "Corticosteroid Hormone Receptor Agonists"
- If you find a compound ID like DB00443, call Feature[DB00443, name] to get the compound name
- If you find a disease ID like DOID:2377, call Feature[DOID:2377, name] to get the disease name
- NEVER finish with raw node IDs - this is a CRITICAL ERROR
- ALWAYS convert node IDs to readable names using Feature function

Here are some examples:
{examples}
(END OF EXAMPLES)
When last Observation has been given or there is no Thought, you should provide next only one Thought based on the question. When last Thought has been given, you should provide next only one Action. When you think it's time to finish, use Finish to end the process. Don't make Observation.

You can use the following actions: Retrieve[<keyword>], Feature[<Node, Feature>], Degree[<Node, Type>], Neighbor[<Node, Type>], SolveQuestion[<question>], SolvePharmacologicClass[<condition>], SolveCellularComponent[<condition, regulation>], SolvePathway[<condition, regulation>], SolveGeneCount[<gene>], SolveDiseaseCount[<disease>], and Finish[<answer>]. The new text you generate should be in a line and less than 512 tokens.
Definition of the graph: {graph_definition}
Question: {question} Please answer by providing node main feature (e.g., names) rather than node IDs. {scratchpad}"""

REFLECT_GraphAgent_INSTRUCTION_COMPOUND = """Solve a question answering task with interleaving Thought, Interaction with Graph, Feedback from Graph steps. In Thought step, you can think about what further information is needed, and In Interaction step, you can get feedback from graphs with four functions: 
(1) Retrieve[keyword], which retrieves the related node from the graph according to the corresponding query.
(2) Feature[Node, feature], which returns the detailed attribute information of Node regarding the given "feature" key.
(3) Degree[Node, neighbor_type], which calculates the number of "neighbor_type" neighbors of the node Node in the graph.
(4) Neighbor[Node, neighbor_type], which lists the "neighbor_type" neighbours of the node Node in the graph and returns them.
Besides, you can use compound function, such as Feature[Retrieve[keyword], feature], which returns the detailed attribute information of Retrieve[keyword] regarding the given "feature" key. HOWEVER, this is DEPRECATED - use direct node IDs instead.
Here are some examples:
{examples}
(END OF EXAMPLES)
{reflections}
When last Observation has been given or there is no Thought, you should provide next only one Thought based on the question. When last Thought has been given, you should provide next only one Action. When you think it's time to finish, use Finish to end the process. Don't make Observation.
Definition of the graph: {graph_definition}
Question: {question} Please answer by providing node main feature (e.g., names) rather than node IDs. {scratchpad}"""


GraphAgent_INSTRUCTION_ZeroShot = """Solve a question answering task with interleaving Thought, Interaction with Graph, Feedback from Graph steps. In Thought step, you can think about what further information is needed, and In Interaction step, you can get feedback from graphs with four functions: 
(1) Retrieve[keyword], which retrieves the related node from the graph according to the corresponding query.
(2) Feature[Node, feature], which returns the detailed attribute information of Node regarding the given "feature" key.
(3) Degree[Node, neighbor_type], which calculates the number of "neighbor_type" neighbors of the node Node in the graph.
(4) Neighbor[Node, neighbor_type], which lists the "neighbor_type" neighbours of the node Node in the graph and returns them.
Definition of the graph: {graph_definition}
Question: {question} Please answer by providing node main feature (e.g., names) rather than node IDs. {scratchpad}"""

REFLECT_GraphAgent_INSTRUCTION = """Solve a question answering task with interleaving Thought, Interaction with Graph, Feedback from Graph steps. In Thought step, you can think about what further information is needed, and In Interaction step, you can get feedback from graphs with four functions: 
(1) Retrieve[keyword], which retrieves the related node from the graph according to the corresponding query.
(2) Feature[Node, feature], which returns the detailed attribute information of Node regarding the given "feature" key.
(3) Degree[Node, neighbor_type], which calculates the number of "neighbor_type" neighbors of the node Node in the graph.
(4) Neighbor[Node, neighbor_type], which lists the "neighbor_type" neighbours of the node Node in the graph and returns them.
Here are some examples:
{examples}
(END OF EXAMPLES)
{reflections}
Definition of the graph: {graph_definition}
Question: {question} Please answer by providing node main feature (e.g., names) rather than node IDs. {scratchpad}"""

REFLECT_GraphAgent_INSTRUCTION_ZeroShot = """Solve a question answering task with interleaving Thought, Interaction with Graph, Feedback from Graph steps. In Thought step, you can think about what further information is needed, and In Interaction step, you can get feedback from graphs with four functions: 
(1) Retrieve[keyword], which retrieves the related node from the graph according to the corresponding query.
(2) Feature[Node, feature], which returns the detailed attribute information of Node regarding the given "feature" key.
(3) Degree[Node, neighbor_type], which calculates the number of "neighbor_type" neighbors of the node Node in the graph.
(4) Neighbor[Node, neighbor_type], which lists the "neighbor_type" neighbours of the node Node in the graph and returns them.
{reflections}
Definition of the graph: {graph_definition}
Question: {question} Please answer by providing node main feature (e.g., names) rather than node IDs. IMPORTANT: When using Finish[], provide ONLY the direct answer without any explanations, prefixes, or additional text. For example, instead of "The answer is X, Y, Z" or "Based on the data, X, Y, Z", just write "X, Y, Z". {scratchpad}"""


REFLECT_Agent_INSTRUCTION_BASE = """You are an advanced reasoning agent that can improve based on self refection. You will be given a previous reasoning trial in which you were given access to specified graph function tools and a question to answer. You were unsuccessful in answering the question either because you guessed the wrong answer with Finish[], you used up your set number of reasoning steps, or the total length of your reasoning is over the limit.  In a few sentences, diagnose a possible reason for failure and devise a new, concise, high level plan that aims to mitigate the same failure. Use complete sentences. 
Here are some examples:
{examples}
(END OF EXAMPLES)
Previous trial:
Solve a question answering task with interleaving Thought, Interaction with Graph, Feedback from Graph steps. In Thought step, you can think about what further information is needed, and in Interaction step, you can get feedback from graphs with four functions: 
(1) Retrieve[keyword], which retrieves the related node from the graph according to the corresponding query.
(2) Feature[Node, feature], which returns the detailed attribute information of Node regarding the given "feature" key.
(3) Degree[Node, neighbor_type], which calculates the number of "neighbor_type" neighbors of the node Node in the graph.
(4) Neighbor[Node, neighbor_type], which lists the "neighbor_type" neighbours of the node Node in the graph and returns them.
Definition of the graph: {graph_definition}
Question: {question}Please answer by providing node main feature (e.g., names) rather than node IDs.{scratchpad}
Reflection:"""

REFLECT_Agent_INSTRUCTION_SHORT_MULTIPLE = """Please write the Reflections, including the content for Recap of the Trial, Guided Reflection, based on the guidance provided in Graph Function Background, Previous Trial Details, Recap of the Trial, and Guided Reflection.
Graph Function Background
- Definition of the graph: {graph_definition}
- You were provided with the following functions to interact with the graph:
  - Retrieve(keyword): Finds the related node based on the query keyword.
  - Feature(Node, feature): Retrieves detailed attribute information for the specified node and feature key.
  - Degree(Node, neighbor_type): Calculates the number of neighbors of the specified type for the given node.
  - Neighbor(Node, neighbor_type): Lists the neighbors of the specified type for the given node.
Recap of the Trial
- Question: [Insert the problem description here]
- Graph Information Used: [List the graph structural information selected in the trial]
- Outcome: The question remained unanswered due to:
  - Incorrect assumptions or guessing.
  - Exhausted reasoning steps or exceeded the reasoning limit.
Guided Reflection
- Understanding the Question
  - Core Goal: What is the main objective of this question?
  - Missed Details: Were any IMPORTANT keywords or relationships overlooked?
  - Misinterpretations: Identify any misunderstandings and correct them.
-  Analysis of Selected Graph Information
  - Relevance: Why did you choose the graph data used? Was it aligned with the goal?
  - Missed Insights: Were there other relevant pieces of information you didn't consider? If so, why?
  - Redundancies: Flag any irrelevant information that added confusion or wasted reasoning steps.
- Align the problem understanding and graph function information
  - Assess whether your understanding of the problem matched the graph selection. Identify inconsistencies and refine your selection criteria to better suit the task.
- Improved Strategy
Based on your reflection:
  - Updated Understanding of the Problem: Revise and describe your updated understanding.
  - Revised Graph Selection: List and explain which graph information you would now choose and why it is more suitable.
  - Avoiding Past Issues: Describe how this strategy addresses the challenges and improves your reasoning.
Here are some examples:
{examples}
(END OF EXAMPLES)
Previous trial:
Question: {question}
{scratchpad}
Reflection:
"""


REFLECT_Agent_INSTRUCTION = """You are an advanced reasoning agent that can improve based on self reflection. Reflect on your prior reasoning trial, and find areas for improvement to enhance your performance in answering the question next time. Please write the Reflections, including the content for Recap of the Trial, Guided Reflection, based on the guidance provided in Graph Function Background, Previous Trial Details, Recap of the Trial, and Guided Reflection.
Graph Function Background
- Definition of the graph: {graph_definition}
- You were provided with the following functions to interact with the graph:
  - Retrieve(keyword): Finds the related node based on the query keyword.
  - Feature(Node, feature): Retrieves detailed attribute information for the specified node and feature key.
  - Degree(Node, neighbor_type): Calculates the number of neighbors of the specified type for the given node.
  - Neighbor(Node, neighbor_type): Lists the neighbors of the specified type for the given node.
Recap of the Trial
- Question: [Insert the problem description here]
- Graph Information Used: [List the graph structural information selected in the trial]
- Outcome: The question was not successfully answered due to one of the following reasons:
  - You guessed the wrong answer with Finish[].
  - You used up the set number of reasoning steps (10 steps).
  - Your total reasoning length exceeded the limit.
Guided Reflection
- Understanding the Question
  - Core Goal: What is the main objective of this question?
  - Missed Information: Could you have overlooked any IMPORTANT details?
  - Potential Misunderstandings: Were there any misinterpretations in your approach? If so, list and correct them.
-  Analysis of Selected Graph Information
  - Relevance: Why did you choose the information you selected? How did it help answer the question?
  - Missed Insights: Were there other relevant pieces of information you didn't consider? If so, why?
  - Redundancies: Did you include irrelevant or redundant information? If yes, identify and revise.
- Align the problem understanding and graph function information
  - Understanding and Selection: How did your understanding of the problem influence your graph structure choice?
  - Inconsistencies: Were there any mismatches between your understanding and your graph structure selection? If so, what caused them?
  - Adjustments: How can you better align your understanding with the graph structure selection?
- Improved Strategy
Based on your reflection:
  - Updated Understanding of the Problem: Revise and describe your updated understanding.
  - Revised Graph Selection: List and explain which graph information you would now choose and why it is more suitable.
  - Avoiding Past Issues: Describe how this strategy addresses the challenges and improves your reasoning.
Here are some examples:
{examples}
(END OF EXAMPLES)
Previous trial Details:
Question: {question} Please answer by providing node main feature (e.g., names) rather than node IDs.
{scratchpad}
Reflection:
"""

COUMPOUND_REFLECT_Agent_INSTRUCTION= """You are an advanced reasoning agent that can improve based on self reflection. Reflect on your prior reasoning trial, and find areas for improvement to enhance your performance in answering the question next time. Please write the Reflections, including the content for Recap of the Trial, Guided Reflection, based on the guidance provided in Graph Function Background, Previous Trial Details, Recap of the Trial, and Guided Reflection.
Graph Function Background
- Definition of the graph: {graph_definition}
- You were provided with the following functions to interact with the graph:
  - Retrieve(keyword): Finds the related node based on the query keyword.
  - Feature(Node, feature): Retrieves detailed attribute information for the specified node and feature key.
  - Degree(Node, neighbor_type): Calculates the number of neighbors of the specified type for the given node.
  - Neighbor(Node, neighbor_type): Lists the neighbors of the specified type for the given node.
  Besides, you can use compound function, such as Feature[Retrieve[keyword], feature], which returns the detailed attribute information of Retrieve[keyword] regarding the given "feature" key.
Recap of the Trial
- Question: [Insert the problem description here]
- Graph Information Used: [List the graph structural information selected in the trial]
- Outcome: The question was not successfully answered due to one of the following reasons:
  - You guessed the wrong answer with Finish[].
  - You used up the set number of reasoning steps (10 steps).
  - Your total reasoning length exceeded the limit.
Guided Reflection
- Understanding the Question
  - Core Goal: What is the main objective of this question?
  - Missed Information: Could you have overlooked any critical details?
  - Potential Misunderstandings: Were there any misinterpretations in your approach? If so, list and correct them.
-  Analysis of Selected Graph Information
  - Relevance: Why did you choose the information you selected? How did it help answer the question?
  - Missed Insights: Were there other relevant pieces of information you didn't consider? If so, why?
  - Redundancies: Did you include irrelevant or redundant information? If yes, identify and revise.
- Align the problem understanding and graph function information
  - Understanding and Selection: How did your understanding of the problem influence your graph structure choice?
  - Inconsistencies: Were there any mismatches between your understanding and your graph structure selection? If so, what caused them?
  - Adjustments: How can you better align your understanding with the graph structure selection?
- Improved Strategy
Based on your reflection:
  - Updated Understanding of the Problem: Revise and describe your updated understanding.
  - Revised Graph Selection: List and explain which graph information you would now choose and why it is more suitable. Explain how these graph functions can be combined into composite functions to streamline operations, ensuring no more than two functions are combined at each step.
  - Avoiding Past Issues: Describe how this strategy addresses the challenges and improves your reasoning.
Here are some examples:
{examples}
(END OF EXAMPLES)
Previous trial Details:
Question: {question} Please answer by providing node main feature (e.g., names) rather than node IDs.
{scratchpad}
Reflection:
"""

COUMPOUND_PLAN_REFLECT_Agent_INSTRUCTION_BASE=""""""
COUMPOUND_PLAN_REFLECT_Agent_INSTRUCTION_SHORT_MULTIPLE=""""""
COUMPOUND_REFLECT_Agent_INSTRUCTION_BASE="""

"""
COUMPOUND_REFLECT_Agent_INSTRUCTION_SHORT_MULTIPLE=""""""

PLAN_GraphAgent_INSTRUCTION = """Solve a question answering task with interleaving Thought, Interaction with Graph, Feedback from Graph steps. In Plan step, you can think about what the question is asking and plan how to do to get the answer. In Thought step, you can think about what further information is needed, and In Interaction step, you can get feedback from graphs with four functions: 
(1) Retrieve[keyword], which retrieves the related node from the graph according to the corresponding query.
(2) Feature[Node, feature], which returns the detailed attribute information of Node regarding the given "feature" key.
(3) Degree[Node, neighbor_type], which calculates the number of "neighbor_type" neighbors of the node Node in the graph.
(4) Neighbor[Node, neighbor_type], which lists the "neighbor_type" neighbours of the node Node in the graph and returns them.
Besides, you can use compound function, such as Feature[Retrieve[keyword], feature], which returns the detailed attribute information of Retrieve[keyword] regarding the given "feature" key.
Here are some examples:
{examples}
(END OF EXAMPLES)
When last Observation has been given or there is no Plan, you should provide next only one Plan based on the question. When last Plan has been given, you should provide next only one Thought. When last Thought has been given, you should provide next only one Action. When you think it's time to finish, use Finish to end the process. Don't make Observation.
Definition of the graph: {graph_definition}
Question: {question} Please answer by providing node main feature (e.g., names) rather than node IDs. {scratchpad}"""

REFLECT_PLAN_GraphAgent_INSTRUCTION = """Solve a question answering task with interleaving Thought, Interaction with Graph, Feedback from Graph steps. In Plan step, you can think about what the question is asking and plan how to do to get the answer. In Thought step, you can think about what further information is needed, and In Interaction step, you can get feedback from graphs with four functions: 
(1) Retrieve[keyword], which retrieves the related node from the graph according to the corresponding query.
(2) Feature[Node, feature], which returns the detailed attribute information of Node regarding the given "feature" key.
(3) Degree[Node, neighbor_type], which calculates the number of "neighbor_type" neighbors of the node Node in the graph.
(4) Neighbor[Node, neighbor_type], which lists the "neighbor_type" neighbours of the node Node in the graph and returns them.
Here are some examples:
{examples}
(END OF EXAMPLES)
{reflections}
When last Observation has been given or there is no Plan, you should provide next only one Plan based on the question. When last Plan has been given, you should provide next only one Thought. When last Thought has been given, you should provide next only one Action. When you think it's time to finish, use Finish to end the process. Don't make Observation.
Definition of the graph: {graph_definition}
Question: {question} Please answer by providing node main feature (e.g., names) rather than node IDs. {scratchpad}"""

REFLECT_PLAN_GraphAgent_NEW_INSTRUCTION = """Solve a question answering task with interleaving Thought, Interaction with Graph, Feedback from Graph steps. In Plan step, you can think about what the question is asking and plan how to do to get the answer. In Thought step, you can think about what further information is needed, and In Interaction step, you can get feedback from graphs with four functions: 
(1) Retrieve[keyword], which retrieves the related node from the graph according to the corresponding query.
(2) Feature[Node, feature], which returns the detailed attribute information of Node regarding the given "feature" key.
(3) Degree[Node, neighbor_type], which calculates the number of "neighbor_type" neighbors of the node Node in the graph.
(4) Neighbor[Node, neighbor_type], which lists the "neighbor_type" neighbours of the node Node in the graph and returns them.
Besides, you can use compound function, such as Feature[Retrieve[keyword], feature], which returns the detailed attribute information of Retrieve[keyword] regarding the given "feature" key.
You can't use any functions other than the four mentioned above. You can't use a combination of more than two functions.
For straightforward questions, prioritize simple actions and avoid unnecessary complexity.
Here are some examples:
{examples}
(END OF EXAMPLES)
{reflections}
When last Observation has been given or there is no Plan, you should provide next only one Plan based on the question. When last Plan has been given, you should provide next only one Thought. When last Thought has been given, you should provide next only one Action. When you think it's time to finish, use Finish to end the process. Don't make Observation.
Definition of the graph: {graph_definition}
Question: {question} Please answer by providing node main feature (e.g., names) rather than node IDs. {scratchpad}"""



EVAL_Agent_INSTRUCTION = """You are an intelligent reasoning accuracy evaluation agent. Evaluate the final answer based on all the thought, action, and observation processes and determine if it meets the problem requirements. Ensure the following: The final answer directly corresponds to the data retrieved from the graph. It satisfies the question's requirement without including irrelevant or incorrect elements. The reasoning behind the answer is logical and supported by the observations. In a few sentences, please provide a brief explanation summarizing why the answer meets or does not meet the criteria. Then, please conclude with a clear judgment based on the  explanation, respond [yes] if the answer is correct , or [no] if the answer is not correct.
Here are some examples:
{examples}
(END OF EXAMPLES)
Solve a question answering task with interleaving Thought, Interaction with Graph, Feedback from Graph steps. In Thought step, you can think about what further information is needed, and In Interaction step, you can get feedback from graphs with four functions: 
(1) Retrieve[keyword], which retrieves the related node from the graph according to the corresponding query.
(2) Feature[Node, feature], which returns the detailed attribute information of Node regarding the given "feature" key.
(3) Degree[Node, neighbor_type], which calculates the number of "neighbor_type" neighbors of the node Node in the graph.
(4) Neighbor[Node, neighbor_type], which lists the "neighbor_type" neighbours of the node Node in the graph and returns them.
Definition of the graph: {graph_definition}
Question: {question}Please answer by providing node main feature (e.g., names) rather than node IDs.{scratchpad}

Proceed with explanation and judgment below:
"""

EVAL_COMPOUND_PLAN_Agent_INSTRUCTION = """You are an intelligent reasoning accuracy evaluation agent. Evaluate the final answer based on all the plan, thought, action, and observation processes and determine if it meets the problem requirements. Ensure the following: The final answer directly corresponds to the data retrieved from the graph. It satisfies the question's requirement without including irrelevant or incorrect elements. The reasoning behind the answer is logical and supported by the observations. In a few sentences, please provide a brief explanation summarizing why the answer meets or does not meet the criteria. Then, please conclude with a clear judgment based on the  explanation, respond [yes] if the answer is correct , or [no] if the answer is not correct.
Here are some examples:
{examples}
(END OF EXAMPLES)
Solve a question answering task with interleaving Thought, Interaction with Graph, Feedback from Graph steps. In Plan step, you can think about what the question is asking and plan how to do to get the answer. In Thought step, you can think about what further information is needed, and In Interaction step, you can get feedback from graphs with four functions: 
(1) Retrieve[keyword], which retrieves the related node from the graph according to the corresponding query.
(2) Feature[Node, feature], which returns the detailed attribute information of Node regarding the given "feature" key.
(3) Degree[Node, neighbor_type], which calculates the number of "neighbor_type" neighbors of the node Node in the graph.
(4) Neighbor[Node, neighbor_type], which lists the "neighbor_type" neighbours of the node Node in the graph and returns them.
Besides, you can use compound function, such as Feature[Retrieve[keyword], feature], which returns the detailed attribute information of Retrieve[keyword] regarding the given "feature" key.
When last Observation has been given or there is no Plan, you should provide next only one Plan based on the question. When last Plan has been given, you should provide next only one Thought. When last Thought has been given, you should provide next only one Action. When you think it's time to finish, use Finish to end the process. Don't make Observation.
Definition of the graph: {graph_definition}
Question: {question}Please answer by providing node main feature (e.g., names) rather than node IDs.{scratchpad}

Proceed with explanation and judgment below:
"""
EVAL_COMPOUND_Agent_INSTRUCTION = """
"""


graph_agent_prompt_zeroshot = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI bot. IMPORTANT: When you use the Finish[] command, please put ONLY the direct answer inside the brackets, without any explanations or additional text."),
    ("human", GraphAgent_INSTRUCTION_ZeroShot),
])

graph_agent_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI bot. Answer step by step. IMPORTANT: When you use the Finish[] command, please put ONLY the direct answer inside the brackets, without any explanations or additional text."),
    ("human", GraphAgent_INSTRUCTION),
])

reflect_graph_agent_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI bot. IMPORTANT: When you use the Finish[] command, please put ONLY the direct answer inside the brackets, without any explanations or additional text."),
    ("human", REFLECT_GraphAgent_INSTRUCTION),
])

reflect_graph_agent_prompt_zeroshot = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI bot. IMPORTANT: When you use the Finish[] command, you must put ONLY the direct answer inside the brackets, without any explanations or additional text."),
    ("human", REFLECT_GraphAgent_INSTRUCTION_ZeroShot),
])

graph_reflect_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI bot. IMPORTANT: When you use the Finish[] command, you must put ONLY the direct answer inside the brackets, without any explanations or additional text."),
    ("human", REFLECT_Agent_INSTRUCTION),
])

graph_reflect_prompt_base = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI bot. IMPORTANT: When you use the Finish[] command, you must put ONLY the direct answer inside the brackets, without any explanations or additional text."),
    ("human", REFLECT_Agent_INSTRUCTION_BASE),
])

graph_reflect_prompt_short_multiple = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI bot. IMPORTANT: When you use the Finish[] command, you must put ONLY the direct answer inside the brackets, without any explanations or additional text."),
    ("human", REFLECT_Agent_INSTRUCTION_SHORT_MULTIPLE),
])

graph_eval_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI bot. IMPORTANT: When you use the Finish[] command, you must put ONLY the direct answer inside the brackets, without any explanations or additional text."),
    ("human", EVAL_Agent_INSTRUCTION),
])

graph_compound_and_plan_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI bot. IMPORTANT: When you use the Finish[] command, you must put ONLY the direct answer inside the brackets, without any explanations or additional text."),
    ("human", PLAN_GraphAgent_INSTRUCTION),
])

graph_compound_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI bot. IMPORTANT: When you use the Finish[] command, you must put ONLY the direct answer inside the brackets, without any explanations or additional text."),
    ("human", GraphAgent_INSTRUCTION_COMPOUND),
])

reflect_graph_compound_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI bot."),
    ("human", REFLECT_GraphAgent_INSTRUCTION_COMPOUND),
])

reflect_graph_compound_and_plan_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI bot."),
    ("human", REFLECT_PLAN_GraphAgent_INSTRUCTION),
])

reflect_graph_compound_and_new_plan_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI bot."),
    ("human", REFLECT_PLAN_GraphAgent_NEW_INSTRUCTION),
])

graph_compound_and_plan_reflect_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI bot."),
    ("human", COUMPOUND_REFLECT_Agent_INSTRUCTION),
])

graph_compound_and_plan_reflect_prompt_base = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI bot."),
    ("human", COUMPOUND_PLAN_REFLECT_Agent_INSTRUCTION_BASE),
])

graph_compound_and_plan_reflect_prompt_short_multiple = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI bot."),
    ("human", COUMPOUND_PLAN_REFLECT_Agent_INSTRUCTION_SHORT_MULTIPLE),
])

graph_compound_reflect_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI bot."),
    ("human", COUMPOUND_REFLECT_Agent_INSTRUCTION),
])


graph_compound_reflect_prompt_base = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI bot."),
    ("human", COUMPOUND_REFLECT_Agent_INSTRUCTION_BASE),
])

graph_compound_reflect_prompt_short_multiple = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI bot."),
    ("human", COUMPOUND_REFLECT_Agent_INSTRUCTION_SHORT_MULTIPLE),
])

graph_compound_and_plan_eval_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI bot."),
    ("human", EVAL_COMPOUND_PLAN_Agent_INSTRUCTION),
])

graph_compound_eval_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI bot."),
    ("human", EVAL_COMPOUND_Agent_INSTRUCTION),
])

# Simple Plan Generation Instruction
SIMPLE_PLAN_GENERATION_INSTRUCTION = """You are an expert at creating step-by-step plans for solving biomedical graph questions.

Given a question and graph definition, create a detailed plan that:
1. Identifies the key entities to find
2. Specifies the graph functions to use
3. Outlines the logical sequence of steps
4. Ensures the final answer is a readable name, not just node IDs

Question: {question}
Graph Definition: {graph_definition}

Create a detailed step-by-step plan:"""

# Plan Evaluation Instructions
PLAN_EVALUATION_INSTRUCTION = """Evaluate the following plan based on these criteria:

1. Correctness (35%): How likely is this plan to lead to the correct answer?
2. Completeness (25%): Does the plan cover all necessary steps?
3. Logical Coherence (20%): Is the plan logically sound and well-structured?
4. Relevance (15%): How relevant is the plan to the question?
5. Efficiency (5%): Is the plan efficient in terms of steps and resources?

For each criterion, give a score from 1-5 (1=poor, 5=excellent).

Plan to evaluate:
{plan}

Question: {question}
Graph definition: {graph_definition}

Evaluation (provide scores and brief justification for each criterion):"""

# Simplified Plan Evaluation with 3 criteria
SIMPLIFIED_PLAN_EVALUATION_INSTRUCTION = """Evaluate the following plan based on these 3 criteria:

1. Logical Coherence (60%): Is the plan logically sound, well-structured, and follows a clear reasoning path?
2. Relevance (20%): How relevant and directly applicable is the plan to the specific question?
3. Completeness (20%): Does the plan cover all necessary steps to reach a comprehensive answer?

For each criterion, give a score from 1-5 (1=poor, 5=excellent).

Plan to evaluate:
{plan}

Question: {question}
Graph definition: {graph_definition}

Evaluation (provide scores and brief justification for each criterion):"""

ANSWER_EVALUATION_INSTRUCTION = """Evaluate the following answer based on these criteria:

1. Accuracy (50%): Does this answer directly address the question and contain relevant, specific information?
2. Completeness (30%): Does it provide comprehensive information that covers the question requirements?
3. Evidence Quality (20%): Was evidence properly retrieved and used from the graph?

EVALUATION GUIDELINES:
- Score 5: Answer is correct, complete, and well-supported by graph evidence
- Score 4: Answer is mostly correct with minor issues
- Score 3: Answer is partially correct but missing important details
- Score 2: Answer has significant errors or is incomplete
- Score 1: Answer is wrong or completely irrelevant

For each criterion, give a score from 1-5 (1=poor, 5=excellent).

IMPORTANT: After providing scores, give your final judgment as either [YES] or [NO] based on the following rule:
- If the total weighted score is 3.0 or higher, output [YES] 
- If the total weighted score is below 3.0, output [NO]
- If the answer is clearly wrong (e.g., "Node 100996379", "Information found in graph but specific answer not determined"), output [NO]
- If the answer contains specific medical terms and addresses the question directly, consider [YES]

Answer to evaluate:
{answer}

Execution process:
{scratchpad}

Question: {question}
Graph definition: {graph_definition}

Evaluation (provide scores and brief justification for each criterion, then final judgment):"""

# Enhanced Graph Counselor Prompt Templates
enhanced_plan_generation_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI bot that generates comprehensive plans."),
    ("human", SIMPLE_PLAN_GENERATION_INSTRUCTION),
])

# New enhanced plan generation using existing complex prompts
enhanced_plan_generation_complex_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI bot that generates detailed step-by-step plans using graph reasoning."),
    ("human", SIMPLE_PLAN_GENERATION_INSTRUCTION),
])

enhanced_plan_evaluation_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI bot that evaluates plans."),
    ("human", SIMPLIFIED_PLAN_EVALUATION_INSTRUCTION),
])

# New simplified plan evaluation with 3 criteria
simplified_plan_evaluation_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI bot that evaluates plans using simplified criteria."),
    ("human", SIMPLIFIED_PLAN_EVALUATION_INSTRUCTION),
])

enhanced_answer_evaluation_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI bot that evaluates answers."),
    ("human", ANSWER_EVALUATION_INSTRUCTION),
])