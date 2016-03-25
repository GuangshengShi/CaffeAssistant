The program "log_parser.py" is created by thesby. And as the main LISENCE, it's under the protection of MIT LISENCE.

#USAGE:
If you have done the training of your caffe model, you should have a log file that contains the output of "LOG".
The constructor of class LogParser can receive a string, while is the path to your log file.  And then, the program will call __parse(). And all the work will be done by itself automatically. After the return of the program, you should have an "LogParser" object.

#EXAMPLE:
```python
parser = LogParser("path/to/your/log")
parser.plot()
```