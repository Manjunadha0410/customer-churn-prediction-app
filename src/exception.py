import sys

def get_exception_details(exception, sys_module):
    tb_obj =  sys_module.exc_info()[2]
    if tb_obj is None:
        return f"Exception occurred: {exception}"
    source_line = tb_obj.tb_lineno
    file_name = tb_obj.tb_frame.f_code.co_filename
    return f"Exception occurred in {file_name} at line {source_line} : {exception}"

class ProjectError(Exception):
    def __init__(self,message,sys_module):
        super().__init__(message)
        self.error_details=get_exception_details(message,sys_module)
    
    def __str__(self):
        return self.error_details
    


        