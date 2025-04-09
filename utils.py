import logging
import os
import shutil


def change_by_coeff(cur_val, min_val, max_val, factor, dempher):
    optFactor = pow(factor, 1. / dempher)
    newvalue = cur_val / optFactor
    if newvalue >= max_val:
        cur_val = max_val
    if newvalue < min_val:
        cur_val = min_val
    else:
        cur_val = newvalue
    return cur_val

def iterate (dim, points, res, current):
    """Функция раскрывает дискретные области значений переменных
    в список точек"""
    if dim >= len(points):
        res.append(current)
        return
    for p_d in points[dim]:
        iterate(dim+1, points, res, current + [p_d])


def clear_dir(dir_path: str):
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            logger = logging.getLogger()
            logger.critical('Failed to delete %s. Reason: %s' % (file_path, e))


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, level):
       self.logger = logger
       self.level = level
       self.linebuf = ''

    def write(self, buf):
       for line in buf.rstrip().splitlines():
          self.logger.log(self.level, line.rstrip())

    def flush(self):
        pass