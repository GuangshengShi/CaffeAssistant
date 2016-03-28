#!/usr/bin/python
# coding=utf-8

import os

class LogParser:
    def __init__(self, log_file):
        self.log_file = log_file
        self.lines = []
        self.solver = ""
        self.net = []
        self.net_dict = []
        self.train_test = []
        self.ind = ""
        if not os.path.exists(log_file):
            raise Exception("ERROR: No such file %s" % log_file)
        self.__parse()

    def __parse(self):
        log_file = open(self.log_file, mode='r')
        content = log_file.read()
        log_file.close()
        if not file:
            raise Exception("ERROR: %s cannot be open" % log_file)
        lines = content.split("\n")
        self.ind = lines[0].split()[0]
        self.lines = lines
        if len(lines) <= 0:
            print("Empty file!")
            return 0
        self.solver, line_id = self.__parse_solver()
        train_net, line_id = self.__parse_net(line_id)
        self.net.append(train_net)
        test_net, line_id = self.__parse_net(line_id)
        self.net.append(test_net)
        train_test = self.__parse_train_test(line_id)
        self.train_test.append(train_test)
        self.net_dict.append(self.__extract_net(self.net[0]))  #Train
        self.net_dict.append(self.__extract_net(self.net[1]))  #Test

    def __parse_solver(self, start_line=0):
        lines = self.lines
        solve_start = False
        solve_end = False
        i = start_line
        solver = ""
        while i < len(lines):
            if lines[i].count(r"Using "):
                solve_start = True
            if solve_start and lines[i].count(r"net:"):
                solve_end = True
                break
            if solve_start and not solve_end:
                solver += lines[i] + " \n"
            i += 1
        return solver, i-1

    def __parse_net(self, start_line=0):
        lines = self.lines
        start = False
        end = False
        i = start_line
        net = ""
        while i < len(lines):
            if lines[i].count(r"name:"):
                start = True
            if start and lines[i].count(self.ind):
                end = True
                break
            if start and not end:
                net += lines[i] + " \n"
            i += 1
        return net, i-1

    '''
    __parse_train_test: parse the text of log
    '''
    def __parse_train_test(self, start_line):
        lines = self.lines
        start = False
        end = False
        i = start_line
        train_test = ""
        train_id = 0
        test_id = 0
        train = []
        test = []
        while i < len(lines):
            if lines[i].count(r"Iteration 0"):
                start = True
            if start and lines[i].count("Optimization Done"):
                end = True
                break
            if start and not end:
                train_test += lines[i] + " \n"
                line = lines[i]
                if line.count("Iteration") and line.count("Test"):
                    current = []
                    iter_id = line.index("Iteration") + len("Iteration") + 1
                    sub_string = line[iter_id:]
                    iter_num = sub_string.split()[0].strip()
                    iter_num = iter_num[0:-1]
                    current.append(iter_num)
                    i += 1
                    if i >= len(lines):
                        break
                    line = lines[i]
                    acc_ind = line.find("accuracy =")
                    if acc_ind > 0:
                        pass
                    else:
                        while line.find("accuracy =") < 0:
                            i += 1
                            if i < len(lines):
                                line = lines[i]
                            else:
                                break
                        if i < len(lines):
                            line = lines[i]
                        else:
                            break
                        acc_ind = line.find("accuracy =")
                    acc_ind += len("accuracy =")
                    sub_string = line[acc_ind:]
                    acc_num = sub_string.split()[0]
                    acc_num.strip()
                    current.append(acc_num)
                    #############################################
                    i += 1
                    if i >= len(lines):
                        break
                    line = lines[i]
                    loss_ind = line.find("loss =")
                    if loss_ind > 0:
                        loss_ind += len("loss =")
                    else:
                        i += 1
                        line = lines[i]
                    loss_num = line[loss_ind:]
                    loss_num = loss_num.split()[0]
                    loss_num.strip()
                    current.append(loss_num)
                    if len(current) == 3:
                        test.append(current)

                elif line.count("Train net output"):
                    current = [0]
                    iter_id = line.find("loss = ") + len("loss = ")
                    sub_string = line[iter_id:]
                    loss_num = sub_string.split()[0]
                    loss_num = loss_num[0:-1]
                    current.append(loss_num)
                    i += 1
                    if i < len(lines):
                        line = lines[i]
                    else:
                        break
                    while line.find("lr = ") < 0:
                        i += 1
                        if i < len(lines):
                            line = lines[i]
                        else:
                            break
                    iter_id = line.find("Iteration ")
                    iter_id += len("Iteration ")
                    iter_num = line[iter_id:].split()[0]
                    current[0] = iter_num[0:-1]
                    lr_id = line.find("lr = ")
                    lr_id += len("lr = ")
                    lr_num = line[lr_id:].split()[0]
                    current.append(lr_num)
                    if len(current) == 3:
                        train.append(current)
            i += 1
        self.train_test.append(train)
        self.train_test.append(test)
        return train_test, i-1

    '''
    plot: show the result of training and testing with a figure
    '''
    def plot(self, show_accuracy=True, show_loss=True):
        if len(self.train_test) < 2:
            raise Exception("ERROR: No data to plot!")
        import matplotlib.pyplot as plt
        if show_loss:
            plt.figure()
            #plt.subplot(211)
            train = self.train_test[0]
            train_iteration = [i[0] for i in train]
            train_loss = [i[1] for i in train]
            train_lr = [i[2] for i in train]
            plt.plot(train_iteration, train_loss, 'b')
            plt.title("Loss")
            plt.xlabel("Iteration")
            plt.ylabel("Loss")
            # plt.subplot(212)
            # plt.plot(train_iteration, train_lr, 'r')
        if show_accuracy:
            plt.figure()
            test = self.train_test[1]
            test_iter = [i[0] for i in test]
            test_acc = [i[1] for i in test]
            plt.plot(test_iter, test_acc)
            plt.title("Accuracy")
            plt.xlabel("Iteration")
            plt.ylabel("Accuracy")
        plt.show()

    '''
        __extract_net: extract net structure and transform it to a dict
    '''
    def __extract_net(self, net=""):
        def find_name(string, loc):
            inner_done = False
            name = ""
            if net[loc] == " ":
                start = False
            else:
                start = True
            while not inner_done:
                # print net[loc]
                if net[loc] == " " and start == False:
                    pass
                elif net[loc] == " " and start == True:
                    inner_done = True
                else:
                    start = True
                    name = net[loc] + name
                loc -= 1
            name = name.strip()
            return name, loc-1
        net_dict = {}
        net_dict["layer"] = []
        if cmp(net, "") == 0:
            net = self.net[0]
        lines = net.split("\n")
        if lines[0].count("name:"):
            net_dict["name"] = lines[0].split(":")[1]
        else:
            raise Exception("ERROR: Unkown net! A net should be start with 'name:'.")
        left_brace_loc = []
        left_brace = []
        right_brace_loc = []
        for i in range(len(net)):
            if net[i] == '{':
                left_brace_loc.append(i)
                left_brace.append(i)
            elif net[i] == '}':
                right_brace_loc.append(i)
                loc = left_brace.pop()
                string = net[(loc+1): (i -1)]
                string = string.strip()
                if len(left_brace)==0:
                    name, location = find_name(net, loc-1)
                    if name == "layer":
                        # unit = {}
                        # unit[name] = self.__extract_layer(string)
                        net_dict["layer"].append(self.__extract_layer(string))
                    else:
                        unit={}
                        words = string.split(":")
                        i = 0
                        while i < len(words):
                            unit[words[i].strip()] = words[i+1].strip()
                            i += 2
                        net_dict[name] = unit
        return net_dict

    '''
    __extract_layer: a private function called by __extract_net to extract each layer
    '''
    def __extract_layer(self, _layer=""):
        ret_layer = {}
        layer = ""
        i = 0
        while i < len(_layer):
            w = _layer[i]
            if w == "{":
                layer += " { "
            elif w == "}":
                layer += " } "
            elif w == ":":
                j = i
                while j >= 0:
                    if _layer[j] == " ":
                        layer = layer[0:-1]
                    else:
                        break
                    j -= 1
                layer += ":"
            else:
                layer += w
            i += 1
        words = layer.split()
        i = 0
        while i < len(words):
            item = words[i].strip()
            if item.count(":"):
                ret_layer[item[0: -1]] = words[i+1].strip()
                i += 2
            elif item == "{":
                param_dict = {}
                param_name = words[i - 1]
                while True:
                    item = words[i].strip()
                    if item == "}":
                        break
                    else:
                        if item.count(":"):
                            param_dict[item[0:-1]] = words[i+1].strip()
                            i += 2
                        else:
                            i += 1
                ret_layer[param_name] = param_dict
            else:
                i += 1

        return ret_layer

    def __string_to_dict(self, string=""):
        string = string.strip()
        if string.count(":") == 0:
            return None
        if string.count("{") > 0 or string.count("}") > 0:
            raise Exception("ERROR: Unkonw net!")
        result = {}
        items = string.split(":")
        if len(items) % 2 != 0:
            raise Exception("ERROR: Unkonw net!")
        i = 0
        while i < len(items):
            result[items[i].strip()] = items[i+1].strip()
            i += 2
        return result

    def show_net(self, input_image_size=[0, 0]):
        #TODO: a figure to show the net
        # if len(input_image_size) != 2 or len(input_image_size) != 3:
        #     raise Exception("ERROR: image size must be 2d or 3d")
        # img_size = []
        # img_size.append(input_image_size)
        pass

if __name__ == "__main__":
    import sys
    parser = LogParser(sys.argv[1])
    parser.plot()
    pass
