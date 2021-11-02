from .base import *
import dj_database_url
db_from_env = dj_database_url.config(conn_max_age=600)
DATABASES['default'].update(db_from_env)

SECRET_KEY = os.environ.get('SECRET_KEY')

DEBUG = False

ALLOWED_HOSTS = ['tryonapps.herokuapp.com']

DATABASES = {
    'default': dj_database_url.config()
}
