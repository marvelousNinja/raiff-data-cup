from datetime import datetime

def generate_model_name(model, score):
    timestr = datetime.utcnow().strftime('%Y%m%d_%H%M')
    model_name = type(model).__name__
    return f'{model_name}-{timestr}-{score:.5f}.model'
