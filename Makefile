quality:
	ruff check src/int_chatbot scripts/
	ruff format --check src/int_chatbot scripts/

style:
	ruff check src/int_chatbot scripts/ --fix
	ruff format src/int_chatbot scripts/
