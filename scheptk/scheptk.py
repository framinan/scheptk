from abc import ABC, abstractmethod #abstract classes
from util import read_tag, print_tag_value, print_tag_vector

class Instance(ABC):
 
    def __init__(self):
        self.jobs = 0 
        self.pt = []
        self.dd = []
        self.w = [] # weigths
        self.r = [] # release dates
    
    # abstract method completion times
    @abstractmethod
    def ct(self, sequence):
        pass
        
    def read_basic_data(self, filename):
        # jobs (mandatory data)
        self.jobs = read_tag(filename,"JOBS")
        self.pt = read_tag(filename,"PT")
        self.w = read_tag(filename, "W")
        # weights (if not, default weights)
        if(self.w ==-1):
            self.w = [1.0 for i in range(self.jobs)]
            print("No weights specified for the jobs. All assummed to be 1.0.")    

        # due dates (if not,  -1 is assumed)
        self.dd = read_tag(filename, "DD")
        if(self.dd ==-1):
            print("No due dates specified for the jobs. All assummed to be infinite.")    
        
        # release dates (if not, 0 is assumed)
        self.r = read_tag(filename,"R")
        if(self.r==-1):
            self.r = [0 for i in range(self.jobs)]
            print("No release dates specified for the jobs. All assummed to be zero.")    

    # print basic data
    def print_basic_data(self):
        print_tag_value("JOBS", self.jobs)
        print_tag_vector("W", self.w)
        print_tag_vector("PT", self.pt)
        #print("[W=")
        #print(self.w)
   
    # concrete method makespan
    def Cmax(self, sequence):
        return max(self.ct(sequence))

    # sum of completion tme
    def SumCj(self, sequence):
        return sum(self.ct(sequence))

    # sum earliness
    def SumEj(self, sequence):
        if(self.dd == -1):
            print("The instance does not have due dates and due-date related objective cannot be computed. The function will return -1")
            return -1
        else:
            earliness = []
            completion_times = self.ct(sequence)
            for index, item in enumerate(completion_times):
                earliness.append(max(item- self.dd[sequence[index]],0))

        return sum(earliness)

    # max earliness
    def Emax(self, sequence):
        if(self.dd == -1):
            print("The instance does not have due dates and due-date related objective cannot be computed. The function will return -1")
            return -1
        else:
            earliness = []
            completion_times = self.ct(sequence)
            for index, item in enumerate(completion_times):
                earliness.append(max(self.dd[sequence[index]]-item,0))

        return max(earliness)


    # sum earliness
    def SumEj(self, sequence):
        if(self.dd == -1):
            print("The instance does not have due dates and due-date related objective cannot be cmputed.")
        else:
            tardiness = []
            completion_times = self.ct(sequence)
            for index, item in enumerate(completion_times):
                tardiness.append(max(self.dd[sequence[index]]-item,0))

        return sum(tardiness)

    # sum tardiness
    def SumTj(self, sequence):
        if(self.dd == -1):
            print("The instance does not have due dates and due-date related objective cannot be cmputed.")
        else:
            tardiness = []
            completion_times = self.ct(sequence)
            for index, item in enumerate(completion_times):
                tardiness.append(max(item- self.dd[sequence[index]],0))

        return sum(tardiness)

    # weighted sum of completion times
    def WjCj(self, sequence):
        wjcj = 0
        ct = self.ct(sequence)
        for j in range(len(sequence)):
            wjcj = wjcj + ct[j] * self.w[sequence[j]]
        return wjcj
