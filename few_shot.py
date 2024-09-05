### PROMPTING
abraham_lincoln_facts = [
    "Abraham Lincoln served as the 16th President of the United States, from 1861 to 1865.",
    "He led the country through its Civil War, preserved the Union, and ended slavery with the Emancipation Proclamation.",
    "Lincoln delivered the Gettysburg Address on November 19, 1863, one of the most famous speeches in American history.",
    "He was born in a log cabin in Hardin County, Kentucky, on February 12, 1809, and grew up in a poor family.",
    "Lincoln was largely self-educated and became a successful lawyer before entering politics.",
    "He was assassinated by John Wilkes Booth at Ford's Theatre in Washington, D.C., on April 14, 1865.",
    "Lincoln's leadership and statesmanship have made him one of the most revered presidents in U.S. history.",
    "He is often referred to as 'Honest Abe' due to his reputation for integrity and honesty.",
    "Abraham Lincoln was 6 feet 4 inches tall, making him the tallest U.S. president."
]


marie_curie_facts = [
    "Marie Curie was a pioneering physicist and chemist, best known for her research on radioactivity.",
    "She was the first woman to win a Nobel Prize, receiving the Nobel Prize in Physics in 1903, and later won a second Nobel Prize in Chemistry in 1911.",
    "Curie discovered the radioactive elements polonium and radium, and her work laid the foundation for the development of X-ray machines and cancer therapy.",
    "She was born in Warsaw, Poland, on November 7, 1867, and later moved to France to pursue her scientific studies.",
    "Curie faced discrimination as a woman in the male-dominated field of science but persevered and became one of the most celebrated scientists of her time.",
    "She died on July 4, 1934, from complications related to her long-term exposure to radiation.",
    "Curie's legacy continues to inspire generations of scientists, and she remains a symbol of women's achievement in STEM fields.",
    "In 1995, Curie became the first woman to be entombed on her own merits in the Panthéon in Paris.",
    "Marie Curie conducted her research in a shed that had no proper ventilation, and she transported tons of raw materials by bicycle because she could not afford the train fare."
]


mahatma_gandhi_facts = [
    "Mahatma Gandhi was born on October 2, 1869, in Porbandar, a coastal town in British-ruled India.",
    "He studied law at University College London and was admitted to the bar in 1891.",
    "Gandhi's activism began in South Africa in 1893, where he spent 21 years advocating for the rights of Indian immigrants.",
    "In 1915, Gandhi returned to India and joined the Indian National Congress, becoming its leader in 1921.",
    "He led the Non-Cooperation Movement in 1920, urging Indians to boycott British goods and institutions.",
    "The Salt March, a protest against the British salt monopoly, took place in 1930, with Gandhi leading a 240-mile trek to the Arabian Sea.",
    "Gandhi was imprisoned multiple times during his lifetime, including for his involvement in the Quit India Movement in 1942.",
    "On January 30, 1948, Mahatma Gandhi was assassinated by Nathuram Godse, a Hindu nationalist, in New Delhi.",
    "His principles of nonviolent resistance, known as Satyagraha, continue to influence movements for social justice worldwide."
]


facts = [abraham_lincoln_facts, marie_curie_facts, mahatma_gandhi_facts]
data_all = []
extract_chain = get_extraction_chain(llm4, my_prompting)
for fact in facts:
  data_fact = extract_chain.invoke(fact)['function']
  data_all.append(data_fact)

my_prompting_gpt3 =  f"""# Knowledge Graph Instructions for GPT-4
You will be given a list of fact. Consider every fact from it to construct a knowledge graph based on that.
 Do not repeat nodes and if a person is referred to in different pronouns or manners (he...) consider it only one node.
 Prioritize relationships over Property keys. If a node has a lot of properties, Avoid Property keys and replace them with relationships. It's a must
  try to make them relationships and reduce the number of preperties of a node as much as you can.
  Do not be lazy, from each fact I provide and from each word, try to extract meaningful relationships
  The following are examples for you to learn from. For each list of facts, you have to generate a knowledge graph like the following:
  {facts[0]}: \n
  The generated knowledge graph is: {data_all[0]}\n
  {facts[1]}: \n
  The generated knowledge graph is: {data_all[1]}\n
  {facts[2]}: \n
  The generated knowledge graph is: {data_all[2]}\n
  Now, it's your turn to do like what has been done:
          """
