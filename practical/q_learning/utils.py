import csv
import logging
import logging.config

logger = logging.getLogger(__name__)

status = {'info': logging.INFO,
          'debug': logging.DEBUG}


def make_logger(log_path='logs.txt', log_status='info'):

    log_status = status[log_status]

    logger = logging.getLogger(__name__)
    logging.config.dictConfig({
            'version': 1,
            'disable_existing_loggers': False,  
            'formatters': {'standard': {'format': '%(asctime)s [%(levelname)s]%(name)s: %(message)s'}},

            'handlers': {'console': {'level': log_status,
                                     'class': 'logging.StreamHandler',
                                     'formatter': 'standard'},

                         'file': {'class': 'logging.FileHandler',
                                  'level': 'DEBUG',
                                  'filename': log_path,
                                  'mode': 'w',
                                  'formatter': 'standard'}, },

            'loggers': {'': {'handlers': ['console', 'file', ],
                             'level': 'DEBUG',
                             'propagate': True}}})

    return logger

    
def save_args(dictionary, path):
    """

    """
    with open(path, 'w') as outfile:
        writer = csv.writer(outfile)

        for k, v in dictionary.items():
            writer.writerow([k]+[v])


class Timer(object):
    """
    Logs useful infomation and times experiments
    """
    def __init__(self):
        self.start_time = time.time()
        self.logger_timer = logging.getLogger('Timer')

    def report(self, output_dict):
        """
        The main functionality of this class
        """
        output_dict['run time'] = self.calc_time()

        log = ['{} : {}'.format(k, v) for k,v in output_dict.items()]
        self.logger_timer.info(log)

    def calc_time(self):
        return (time.time() - self.start_time) / 60
