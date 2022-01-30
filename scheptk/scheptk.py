from abc import ABC, abstractmethod #abstract classes
#import ast # to get the proper data type
import sys # to stop the exectuion (funcion exit() )


from scheptk.util import print_tag_value, print_tag_vector, get_proper_type, read_tag


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
        # if jobs = -1 the program cannot continue
        if(self.jobs ==-1):
            print("No jobs specified. The program cannot continue.")
            sys.exit()
        # processing times (mandatory data)
        self.pt = read_tag(filename,"PT")
        if(self.pt ==-1):
            print("No processing times specified. The program cannot continue.")
            sys.exit()
        else:
            if(len(self.pt) != self.jobs ):
                print("Number of processing times does not match the number of jobs (JOBS={}, length of PT={}). The program cannot continue".format(self.jobs, len(self.pt)) )
                sys.exit()                
        # weights (if not, default weights)
        self.w = read_tag(filename, "W")
        if(self.w ==-1):
            self.w = [1.0 for i in range(self.jobs)]
            print("No weights specified for the jobs. All weights set to 1.0.")    
        # due dates (if not,  -1 is assumed)
        self.dd = read_tag(filename, "DD")
        if(self.dd ==-1):
            print("No due dates specified for the jobs. All due dates assummed to be infinite.")           
        # release dates (if not, 0 is assumed)
        self.r = read_tag(filename,"R")
        if(self.r==-1):
            self.r = [0 for i in range(self.jobs)]
            print("No release dates specified for the jobs. All release dates set to zero.")    

    # print basic data
    def print_basic_data(self):
        print_tag_value("JOBS", self.jobs)
        print_tag_vector("PT", self.pt)
        print_tag_vector("DD", self.dd)
        print_tag_vector("R", self.r)
        print_tag_vector("W", self.w)

   
    # concrete method makespan
    def Cmax(self, sequence):
        return max(self.ct(sequence))

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

    # max lateness
    def Lmax(self, sequence):
        if(self.dd == -1):
            print("The instance does not have due dates and due-date related objective cannot be cmputed.")
        else:
            lateness = []
            completion_times = self.ct(sequence)
            for index, item in enumerate(completion_times):
                lateness.append(item-self.dd[sequence[index]])
        
        return max(lateness)

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

    # sum lateness
    def SumLj(self, sequence):
        if(self.dd == -1):
            print("The instance does not have due dates and due-date related objective cannot be computed.")
        else:
            lateness = []
            completion_times = self.ct(sequence)
            for index, item in enumerate(completion_times):
                lateness.append(item-self.dd[sequence[index]])
        
        return sum(lateness)

    # sum tardiness
    def SumTj(self, sequence):
        if(self.dd == -1):
            print("The instance does not have due dates and due-date related objective cannot be computed.")
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


# class to implement the single machine layout
class SingleMachine(Instance):
  
   
    def __init__(self, filename):
        print("----- Reading SingleMachine instance data from file " + filename + " -------")
        self.read_basic_data(filename)
        self.print_basic_data()
        print("----- end of SingleMachine instance data from file " + filename + " -------")
        


    def ct(self,sequence):
        completion_time = []
        completion_time.append(self.r[sequence[0]] + self.pt[sequence[0]])
        for i in range(1,len(sequence)):
            completion_time.append(max(completion_time[i-1],self.r[sequence[i]]) + self.pt[sequence[i]])
        return completion_time

    



       