## 1. LOGS EN PYTHON

Para poder hacer logs en python, se puede usar el modulo `logging`. Este modulo permite hacer log de mensajes en diferentes niveles de severidad. Los niveles de severidad son los siguientes:

- DEBUG: Mensajes de depuración. Hay que marcar lo que queramos depurar con este nivel. Normalmente es para info extensa que no queremos que se muestre en producción.
- INFO: Mensajes informativos. Son mensajes que queremos que se muestren en producción. Funcionamiento normal del programa.
- WARNING: Mensajes de advertencia. Son mensajes que no son errores pero que pueden ser importantes.
- ERROR: Mensajes de error. Son mensajes que indican que algo ha ido mal.
- CRITICAL: Mensajes críticos. Son mensajes que indican que algo ha ido muy mal tal que el programa no puede continuar o no de forma segura o esperada.

Para poder hacer logs en python, se debe de seguir los siguientes pasos:

1. Importar el modulo `logging`.

```python
import logging
```

2. Configurar el modulo `logging`:

```python
logger = logging.getLogger('DATA PIPELINE') # Se crea un logger con el nombre 'DATA PIPELINE'
logger.setLevel(logging.DEBUG) # Se establece el nivel de severidad del logger, todos los mensajes con nivel de severidad mayor o igual a DEBUG se mostrarán

# Se crea un manejador de archivos para guardar los logs en un archivo
current_dir = os.path.dirname(os.path.realpath(__file__)) # directorio actual
data_dir = os.path.join('data') # directorio donde se guardarán los logs
file_handler = logging.FileHandler(f'{data_dir}/pipeline_outputs/log.txt') # se define el manejador de archivos

file_handler.setLevel(logging.INFO) # se establece el nivel de severidad del manejador de archivos, todos los mensajes con nivel de severidad mayor o igual a INFO se guardarán en el archivo

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s') # creamos un formatter para los mensajes de log, define cómo se mostrará la información
file_handler.setFormatter(formatter)

# Añadir el manejador de archivos al logger
logger.addHandler(file_handler)
```

3. Podemos hacer que el logger recoja los prints de la consola:

```python
# Redeifinir sys.stdout para que los prints se guarden en el logger
class PrintLogger:
    def __init__(self, logger):
        self.logger = logger

    def write(self, message):
        if message.strip():  # Evitar logs vacíos
            self.logger.info(message.strip())

    def flush(self):
        pass  # Requerido para que funcione sys.stdout

sys.stdout = PrintLogger(logger) # Redefinir sys.stdout para que los prints se guarden en el logger
```

4. Hacer logs:

```python
logger.debug("Este es un mensaje de DEBUG.")
logger.info("Este es un mensaje de INFO.")
logger.warning("Este es un mensaje de WARNING.")
logger.error("Este es un mensaje de ERROR.")
logger.critical("Este es un mensaje de CRITICAL.")
```	    

5. Podemos crear un objeto que cree decoradores para hacer logs:

```python
from datetime import datetime

class LogHeaders:
    def __init__(self, logger):
        self.logger = logger
    def log_header(self, message: str = 'Start of log', date: bool = True):
        self.logger.info('#'*100)
        date = datetime.now().strftime('%Y-%m-%d-%H-%M')
        message = f'{date}    -    {message}' if date else message
        # calculate the number of spaces to center the message
        spaces = int((100 - len(message))/2) - 1
        self.logger.info(f'#{" "*spaces}{message}{" "*spaces}#')       
        self.logger.info('#'*100)

    def log_footer(self, message: str = 'End of log'):
        self.logger.info('-'*100)
        # calculate the number of spaces to center the message
        spaces = int((100 - len(message))/2) - 1
        self.logger.info(f'#{" "*spaces}{message}{" "*spaces}#')       
        self.logger.info('_'*100)

    def log_section(self, message: str):
        message = f' SECTION: {message} '
        self.log_header(message)

    def log_bar(self):
        self.logger.info('-'*100)
    
    def log_double_bar(self):
        self.logger.info('='*100)

    def log_subbar(self):
        self.logger.info('_'*100)
```

6. Uso de los decoradores:

```python
log_headers = LogHeaders(logger)

log_headers.log_header('Start of log', date=True)

log_headers.log_section('PIPELINE')

...

log_headers.double_bar()

...

log_headers.log_footer('End of log')
```

Lo que daría como resultado algo como esto:


```bash
2024-07-03 14:20:52,359 - DATA PIPELINE - ####################################################################################################
2024-07-03 14:20:52,359 - DATA PIPELINE - #                              2024-07-03-14-20    -    Start of log                              #
2024-07-03 14:20:52,359 - DATA PIPELINE - ####################################################################################################
2024-07-03 14:20:52,359 - DATA PIPELINE - ----------------------------------------------------------------------------------------------------
2024-07-03 14:20:52,360 - DATA PIPELINE - #                                             PIPELINE                                             #
2024-07-03 14:20:52,360 - DATA PIPELINE - ----------------------------------------------------------------------------------------------------
2024-07-03 14:20:52,360 - DATA PIPELINE - Processing data...
    .           .               .                                                       .               
    .           .               .                                                       .               
    .           .               .                                                       .               
2024-07-03 14:20:55,565 - DATA PIPELINE - ====================================================================================================
2024-07-03 14:20:55,565 - DATA PIPELINE - ELAPSED: 0:00:03.206933
2024-07-03 14:20:55,565 - DATA PIPELINE - ----------------------------------------------------------------------------------------------------
2024-07-03 14:20:55,565 - DATA PIPELINE - #                                            End of log                                            #
2024-07-03 14:20:55,565 - DATA PIPELINE - ____________________________________________________________________________________________________
```



