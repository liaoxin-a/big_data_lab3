# Лабораторная работа  №3
**Студент:** Liao Xin, M4150

**Вариант:** №1	Hashicorp Vault

**Цель работы:**

Получить навыки выгрузки исходных данных и отправки результатов модели с использованием различных источников данных согласно варианту задания.


**Структура каталогов:**
```
C:.
│   .dvcignore
│   .gitignore
│   dataset.dvc
│   docker-compose.yml----------------------  :конфигурация создания контейнера и образа модели;
│   Dockerfile----------------------  :конфигурация создания контейнера и образа модели;
│   requirements.txt----------------------  : используемые зависимости (библиотеки) и их версии;
│
├───.dvc
│   │   .gitignore
│   │   config
│
├───CD
│       Jenkinsfile
│
├───CI
│       Jenkinsfile
│
├───config
│       config.ini: гиперпараметры модели;
│       config.py
│
├───dataset
│       test_features.npy
│       test_ids.csv
│       train_features.npy
│       train_ids.csv
├───mysql
│       databse_test.sql
│       databse_train.sql
│       Dockerfile
│
├───notebooks
│       youtube8m_classfication.ipynb
│
└───src
    │   data.py
    │   model.py
    │   predict.py
    │   train.py
    │
    ├───unit_tests
    │       test_preprocess.py
    │       test_training.py
    │       __init__.py

```

## Задание:

1. Создать репозитории-форк модели на GitHub, созданной в рамках лабораторной работы №2, регулярно проводить commit + push в ветку разработки, важна история коммитов.
2. Настроить хранилище секретов согласно варианту:.
3. Реализовать взаимодействие следующим образом.
4. Создать CI pipeline (Jenkins, Team City, Circle CI и др.) для сборки docker image и отправки его на DockerHub,   сборка должна автоматически стартовать по pull request в основную ветку репозитория модели;
5. Создать CD pipeline для запуска контейнера и проведения функционального тестирования по сценарию, запуск должен стартовать по требованию или расписанию или как вызов с последнего этапа CI pipeline;

## Hashicorp Vault
 
>HashiCorp Vault — инструмент с открытым исходным кодом, который обеспечивает безопасное хранение и доступ к различным секретам (паролям, сертификатам, токенам). Образ приложения содержит предустановленную сборку HashiCorp Vault, которая при помощи Yandex Key Management Service дополнительно поддерживает Auto Unseal .

## Настроить хранилище секретов:

```
# mysql/Dockerfile
docker run -d --rm --name vault-server --cap-add=IPC_LOCK -p 8200:8200 -e 'VAULT_DEV_ROOT_TOKEN_ID=${pass}' -e 'VAULT_DEV_LISTEN_ADDRESS=0.0.0.0:8200' vault
```
run vault-server
![containers](https://github.com/liaoxin-a/big_data_lab3/blob/main/imgs/containers.PNG)
add mysql password in secret path
![valus](https://github.com/liaoxin-a/big_data_lab3/blob/main/imgs/valus.PNG)

## Реализовать взаимодействие: 
```
# src/data.py
#connect HashiCorp Vault
def connect2vault():
    client = hvac.Client(url=hvac_client_url,token=hvac_token)
    print(client.is_authenticated())
    read_response = client.secrets.kv.read_secret_version(path='mysql')
    return read_response

def connect2mysql():
    user_value='root'
    hvac_response=connect2vault()
    password_value=hvac_response['data']['data']['root'] ####get root password from HashiCorp Vault Response
    host_value='mysql'
    database_value= os.getenv("NAME_DATABASE")
    type_connect=False

    try:
        connection = mysql.connector.connect(
            user=user_value, password=password_value, host=host_value, port='3306', database=database_value)
        print("mysql connected")
        type_connect=True
    except:
        return type_connect,None
    return type_connect,connection

```
   
## удалить локальные:
```
  web:
    build: .
    container_name: c2_bigdata
    environment:
      HVAC_CLIENT: 'http://host.docker.internal:8200'
      NAME_DATABASE: 'db'
      # DB_PASS: ${pass}          -----------delect password
      HVAC_CLIENT_TOKEN: ${token}
    command: bash -c "python src/train.py && python src/predict.py -m && coverage run src/unit_tests/test_preprocess.py && coverage run -a src/unit_tests/test_training.py && coverage report -m"
    ports:
      - 8000:8000
    image: liaox1/big_data:3.0
    depends_on:
        - mysql
    extra_hosts:
        - 'host.docker.internal:host-gateway'
```


## CI:
![CI](https://github.com/liaoxin-a/big_data_lab3/blob/main/imgs/CI.PNG)


## docker hub:
![hub](https://github.com/liaoxin-a/big_data_lab3/blob/main/imgs/docker%20hub.PNG)

