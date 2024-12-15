PROMPT_DICT = {
    'summeval-coh-comp': f"Article: <context>\n\nSummary A: <A>\n\nSummary B: <B>\n\nWhich Summary is more coherent, Summary A or Summary B?",
    'summeval-con-comp': f"Article: <context>\n\nSummary A: <A>\n\nSummary B: <B>\n\nWhich Summary is more consistent to the article, Summary A or Summary B?",
    'summeval-flu-comp': f"Article: <context>\n\nSummary A: <A>\n\nSummary B: <B>\n\nWhich Summary is more fluent, Summary A or Summary B?",
    'summeval-rel-comp': f"Article: <context>\n\nSummary A: <A>\n\nSummary B: <B>\n\nWhich Summary is more relevant to the information of the article, Summary A or Summary B?",

    'topicalchat-coh-comp': f"Dialogue: <context>\n\nResponse A: <A>\n\nResponse B: <B>\n\nWhich Response is more coherent, Response A or Response B?",
    'topicalchat-con-comp': f"Dialogue: <context>\n\nResponse A: <A>\n\nResponse B: <B>\n\nWhich Response continues the dialogue better, Response A or Response B?",
    'topicalchat-eng-comp': f"Dialogue: <context>\n\nResponse A: <A>\n\nResponse B: <B>\n\nWhich Response is more engaging, Response A or Response B?",
    'topicalchat-nat-comp': f"Dialogue: <context>\n\nResponse A: <A>\n\nResponse B: <B>\n\nWhich Response appears more natural, Response A or Response B?",
    
    #     'hanna-coh-comp': f"Evaluate the coherency of the following two stories to determine which one is more coherent.\n\nStory A:\n<A>\n\nStory B:\n<B>\n\nWhich story is more coherent, Story A or Story B?",
    #     'hanna-sur-comp': f"Evaluate and compare two stories in terms of their ability to evoke surprise and unexpected plot twist, to determine which one is more surprising.\n\nStory A:\n<A>\n\nStory B:\n<B>\n\nWhich story is more surprising, Story A or Story B?",
    #     'hanna-com-comp': f"Evaluate the narrative complexity of the following creative stories in terms of structure, characters and themes, and determine which one is more complex.\n\nStory A:\n<A>\n\nStory B:\n<B>\n\nWhich story is more complex, Story A or Story B?",
    #'cmcqrd-diff-comp': f"Evaluate the difficulty of the following

    'hanna-coh-comp': f"Story A:\n<A>\n\nStory B:\n<B>\n\nWhich story is more coherent, Story A or Story B?",
    'hanna-sur-comp': f"Story A:\n<A>\n\nStory B:\n<B>\n\nWhich story is more surprising, Story A or Story B?",
    'hanna-com-comp': f"Story A:\n<A>\n\nStory B:\n<B>\n\nWhich story is more complex, Story A or Story B?",
    
    'cmcqrd-dif-comp': f"Question A:\n<A>\n\nQuestion B:\n<B>\n\nWhich reading comprehension question is more difficult to answer, Question A or Question B?",
}