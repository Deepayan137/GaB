payload = {
		"""<s> [INST] <<SYS>> 
		You are an expert annotator for Question Answering task.
		Your job is to form Question Answer pairs from a given image caption. Read the caption very carefully
		and ask questions with respect to various physical attributes, location, count, action etc.

		<</SYS>>

		CAPTION: The image captures a snowboarder in mid-air, performing a jump on a snow-covered slope. 
		The snowboarder is wearing a gray jacket and is surrounded by trees.
		The snowboard is visible beneath the snowboarder, and the scene is filled with excitement and energy.
		{"rec":[{"Q":"What is the snowboarder doing?", "A":"Skiing"}, {"Q":"What is the man wearing?", "A":"Jacker"}],
		"loc":[{"Q":"Where is the snowborader?", "A":"air"}],
		"judge":[{"Q":"Is the snowboarder flying?", "A":"Yes"}],
		"com":[{"Q":"Is the atmosphere filled with excitement", "A":"Yes"}],
		"act":[{"Q":"Is the snowboarder perfroming a jump?", "A":"Yes"}]
		"count":[{"Q":"How many snowboarders are there?", "A":"1"}],
		"color:[{"Q":"What is the color of his jacket", "A":"Gray"}],
		"sub":[{"Q":"", "A":""}],
		"typ":[{"Q":"", "A":""}],
		"cau":[{"Q":"", "A":""}]
		}
		"""
}