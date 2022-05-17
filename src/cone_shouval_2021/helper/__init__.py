def is_etrace(data):
    return hasattr(data, 'etrace') and data.etrace

def log_time(logger, timer_label, experiment_label, start=True, stop=False):
    if start:
        logger.log_timer.start("{} {}".format(timer_label, experiment_label))
    elif stop:
        logger.log_timer.stop("{} {}".format(timer_label, experiment_label))