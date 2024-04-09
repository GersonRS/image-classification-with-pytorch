.PHONY: export-req

export-req:
	poetry export --without-hashes --without-urls | awk '{ print $$1 }' FS=';' > requirements.txt


# Comando para instalar dependências do projeto em modo desenvolvimento a partir do poetry
install-dev-requirements-from-poetry:
	pip3 install poetry==1.8.1
	pip3 install poetry-plugin-export==1.6.0
	poetry export -f requirements.txt --output requirements-temp.txt --without-hashes --without-urls --with dev,test,docs
	pip3 install -r requirements-temp.txt
	rm -r requirements-temp.txt

# Comando para dar o build e instalar o projeto simdiesel
install-simdiesel:
	pip3 install --upgrade pip setuptools wheel
	pip3 install -e .

# Comando para exportar o virtualenv para o kernel usado nos notebooks
install-kernel-notebooks:
	ipython kernel install --name .venv --user

# Comando para desinstalar o kernel usado nos notebooks
uninstall-kernel-notebooks:
	jupyter kernelspec uninstall .venv -y

# Comando para instalar todas as dependências do projeto
install-dev-requirements: install-dev-requirements-from-poetry \
		install-simdiesel \
		install-kernel-notebooks
