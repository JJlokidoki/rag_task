.PHONY: install prepare run

# Установка зависимостей
install:
	pip install -r requirements.txt

# Запуск тестов
prepare:
	python -c 'pass'

run:
	streamlit run web_interface.py