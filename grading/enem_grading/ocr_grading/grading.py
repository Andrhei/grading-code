import os
import json
import argparse
import logging
# import boto3
# from botocore.exceptions import ClientError
from collections import defaultdict

# API URL configurado em variável de ambiente
# URL = os.getenv('URL', 'url')
# # API TOKEN configurado em variável de ambiente
# TOKEN = os.getenv('TOKEN', 'token')
# # Nome da tabela configurado em variável de ambiente
# TABLE_NAME = os.getenv('RESULTS_TABLE', 'table')

# dynamodb = boto3.resource('dynamodb')

def response_status(examId, studentId, stage='correction', status='success', error=None, summary=None, answers=None):
    """
    Gera um dicionário com o status da operação.

    Args:
        examId (str): ID do exame.
        studentId (str): ID do estudante.
        stage (str): Etapa do processo (default: 'correction').
        status (str): Status da operação (default: 'success').
        error (str, optional): Mensagem de erro, se houver.
        summary (dict, optional): Resumo da operação.
        answers (list, optional): Respostas do estudante.

    Returns:
        dict: Dicionário com o status da operação.
    """
    response = {
        "examId": examId,
        "studentId": studentId,
        "stage": stage,
        "status": status,
        "error": error,
        "summary": summary,
        "answers": answers,
    }
    return {key: value for key, value in response.items() if value is not None}

def load_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Erro ao decodificar JSON no arquivo {path}: {e}")

def grade_exam(answersKey, studentAnswers):

    if answersKey.get('examId') == studentAnswers.get('examId'):
    
        k_map = defaultdict(lambda: None, {int(q['questionNumber']): q['answer'].lower() for q in answersKey.get('answersKey', [])})
        a_map = defaultdict(lambda: None, {int(q['questionId']): q['answer'].lower() for q in studentAnswers.get('answers', [])})

        totalQuestions = len(answersKey.get('answersKey'))
        correctAnswers = 0
        details = []

        for questionId, correctQuestion in k_map.items():
            studentAnswer = a_map.get(questionId)
            isTrue = (studentAnswer == correctQuestion)
            if isTrue:
                correctAnswers += 1
            details.append({
                'questionId':       questionId,
                'studentAnswer':    studentAnswer,
                'correctQuestion':  correctQuestion,
                'correct':          isTrue
            })
        
        summary = {
                'totalQuestions':  totalQuestions,
                'correctAnswers':  correctAnswers,
                'wrongAnswers':    totalQuestions - correctAnswers,
            }
        answers = details
        
        return response_status(examId=studentAnswers.get('examId'),
                               studentId=studentAnswers.get('studentId'),
                               summary=summary,
                               answers=answers)
    
    else:
        return response_status(examId=studentAnswers.get('examId'),
                               studentId=studentAnswers.get('studentId'),
                               status="error",
                               error="Exam not identified!")

def save_to_dynamodb(obj):
    try:
        table = dynamodb.Table(TABLE_NAME)
        table.put_item(Item=obj)
    except ClientError as e:
        print(f'Erro ao gravar à correção - {e}')
    except Exception as e:
        print(f'Erro para acessar DynamoDB - {e}')

#-----------------------------------INICIO LAMBDA HANDLER-----------------------------------
# def lambda_handler(event, context):
#     try:
#         for record in event.get('Records', []):
#             try:
#                 # 'body' é a string JSON enviada para a fila
#                 body_str = record['body']
#                 studentAnswers = json.loads(body_str)
#                 # answersKey = ? # Chamada API Solis
#             except json.JSONDecodeError as e:
#                 print(f"Não foi possível decodificar JSON: {e}")
#             try:
#                 response = grade_exam(
#                     answersKey=answersKey,
#                     studentAnswers=studentAnswers
#                 )
#             except Exception as e:
#                 print(f'Erro na correção da prova - {e}')
#             try:
#                 save_to_dynamodb(response)
#             except Exception as e:
#                 print(f'Erro ao gravar à correção - {e}')
#     except Exception as e:
#                 print(f'Erro ao gravar à correção - {e}')
#-----------------------------------FIM LAMBDA HANDLER-----------------------------------

def main(answersKey_path, answers_path, output_path=None):
    logging.basicConfig(level=logging.INFO)
    logging.info("Inicializando o processo de correção...")
    try:
        logging.info("Leitura do gabarito e respostas do aluno...")
        answersKey = load_json(answersKey_path)
        answers = load_json(answers_path)
    except Exception as e:
        print(f'Erro de recuperação dos objetos JSON - {e}')
    try:    
        logging.info("Iniciando a correção da prova...")
        response = grade_exam(
            answersKey=answersKey,  
            studentAnswers=answers
        )
    except Exception as e:
        print(f'Erro na correção da prova - {e}')
    try:
        logging.info("Gravando o resultado da correção...")
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(response, f, ensure_ascii=False, indent=2)
            print(f"Resultado gravado em: {output_path}")
        else:
            print(json.dumps(response, ensure_ascii=False, indent=2))
    except Exception as e:
        print(f'Erro ao gravar à correção - {e}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Corrige uma prova a partir de arquivos JSON locais."
    )
    parser.add_argument(
        '--answersKey-file', '-k', required=True,
        help="Caminho para o JSON do gabarito. Ex.: {\"answersKey\": [{\"questionId\": \"1\", \"answer\": \"a\"}, ...]}"
    )
    parser.add_argument(
        '--answers-file', '-a', required=True,
        help="Caminho para o JSON das respostas do aluno. Ex.: {\"answers\": [{\"questionId\": \"1\", \"answer\": \"b\"}, ...]}"
    )
    parser.add_argument(
        '--output-file', '-o', required=False,
        help="Caminho para salvar o resultado da correção (JSON). Se omitido, imprime no terminal."
    )
    args = parser.parse_args()

    main(args.answersKey_file, args.answers_file, args.output_file)

