from abc import ABC, abstractmethod #abstract classes
import sys # to stop the execution (funcion exit() )


from scheptk.util import print_tag_value, print_tag_vector, print_tag_matrix, read_tag, find_index_min


class Instance(ABC):
 
    # basic data, common to all layouts
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
 
    # concrete method makespan
    def Cmax(self, sequence):
        return max(self.ct(sequence))

    # max earliness
    def Emax(self, sequence):
         if(self.dd == -1):
            print("The instance does not have due dates and due-date related objective cannot be computed.")        
         return max([ max(self.dd[sequence[index]] - item,0) for index,item in enumerate(self.ct(sequence))])

   # max flowtime
    def Fmax(self, sequence):
        return max([item - self.r[sequence[index]] for index,item in enumerate(self.ct(sequence))])
        
    # max lateness
    def Lmax(self, sequence):
        if(self.dd == -1):
            print("The instance does not have due dates and due-date related objective cannot be computed.")
        return max([ item - self.dd[sequence[index]] for index,item in enumerate(self.ct(sequence))])

    # max tardiness
    def Tmax(self, sequence):
        if(self.dd == -1):
            print("The instance does not have due dates and due-date related objective cannot be computed.")
        return max([ max(item - self.dd[sequence[index]],0) for index,item in enumerate(self.ct(sequence))])

    # sum of completion tme
    def SumCj(self, sequence):
        return sum(self.ct(sequence))

    # sum earliness
    def SumEj(self, sequence):
        if(self.dd == -1):
            print("The instance does not have due dates and due-date related objective cannot be computed.")        
        return sum([ max(self.dd[sequence[index]] - item,0) for index,item in enumerate(self.ct(sequence))])

   # sum flowtime
    def SumFj(self, sequence):
        return sum([item - self.r[sequence[index]] for index,item in enumerate(self.ct(sequence))])

    # sum lateness
    def SumLj(self, sequence):
        if(self.dd == -1):
            print("The instance does not have due dates and due-date related objective cannot be computed.")
        return sum([ item - self.dd[sequence[index]] for index,item in enumerate(self.ct(sequence))])

    # sum tardiness
    def SumTj(self, sequence):
        if(self.dd == -1):
            print("The instance does not have due dates and due-date related objective cannot be computed.")
        return sum([ max(item - self.dd[sequence[index]],0) for index,item in enumerate(self.ct(sequence))])           

    # sum of tardy jobs
    def SumUj(self, sequence):
        if(self.dd == -1):
            print("The instance does not have due dates and due-date related objective cannot be computed.")
        return sum([1 if (item - self.dd[sequence[index]]) > 0 else 0 for index,item in enumerate(self.ct(sequence))])

    # weighted makespan
    def WjCmax(self, sequence):
        return max([item * self.w[sequence[index]] for index,item in enumerate(self.ct(sequence))])

    # weighted max earliness
    def WjEmax(self, sequence):
        if(self.dd == -1):
            print("The instance does not have due dates and due-date related objective cannot be computed.")        
        return max([ (max(self.dd[sequence[index]] - item,0) * self.w[sequence[index]]) for index,item in enumerate(self.ct(sequence))])

   # weighted max flowtime
    def WjFmax(self, sequence):
        return max([(item - self.r[sequence[index]])* self.w[sequence[index]] for index,item in enumerate(self.ct(sequence))])
        
    # weighted max lateness
    def WjLmax(self, sequence):
        if(self.dd == -1):
            print("The instance does not have due dates and due-date related objective cannot be computed.")
        return max([ (item - self.dd[sequence[index]])*self.w[sequence[index]] for index,item in enumerate(self.ct(sequence))])

    # weighted max tardiness
    def WjTmax(self, sequence):
        if(self.dd == -1):
            print("The instance does not have due dates and due-date related objective cannot be computed.")
        return max([ (max(item - self.dd[sequence[index]],0) * self.w[sequence[index]]) for index,item in enumerate(self.ct(sequence))])
        
    # weighted sum of completion times
    def SumWjCj(self, sequence):
        return sum([item * self.w[sequence[index]] for index,item in enumerate(self.ct(sequence))])

    # weighted sum of earliness
    def SumWjEj(self, sequence):
        if(self.dd == -1):
            print("The instance does not have due dates and due-date related objective cannot be computed.")        
        return sum([ (max(self.dd[sequence[index]] - item,0) * self.w[sequence[index]]) for index,item in enumerate(self.ct(sequence))])

    # weighted sum of flowtime
    def WjFmax(self, sequence):
        return sum([(item - self.r[sequence[index]])* self.w[sequence[index]] for index,item in enumerate(self.ct(sequence))])
   
    # weighted sum of lateness
    def SumWjLj(self, sequence):
        if(self.dd == -1):
            print("The instance does not have due dates and due-date related objective cannot be computed.")
        return sum([ (item - self.dd[sequence[index]])*self.w[sequence[index]] for index,item in enumerate(self.ct(sequence))])

    # weighted sum of tardiness
    def SumWjTj(self, sequence):
        if(self.dd == -1):
            print("The instance does not have due dates and due-date related objective cannot be computed.")
        return sum([ (max(item - self.dd[sequence[index]],0) * self.w[sequence[index]]) for index,item in enumerate(self.ct(sequence))])

    # weighted sum of tardy jobs
    def SumWjUj(self, sequence):
        if(self.dd == -1):
            print("The instance does not have due dates and due-date related objective cannot be computed.")
        return sum([1 if (item - self.dd[sequence[index]]) > 0 else 0 for index,item in enumerate(self.ct(sequence))])


# class to implement the single machine layout
class SingleMachine(Instance):
     
    def __init__(self, filename):

        print("----- Reading SingleMachine instance data from file " + filename + " -------")
        # jobs (mandatory data)
        self.jobs = read_tag(filename,"JOBS")
        # if jobs = -1 the program cannot continue
        if(self.jobs ==-1):
            print("No jobs specified. The program cannot continue.")
            sys.exit()
        else:
            print_tag_value("JOBS", self.jobs)
        
        # processing times (mandatory data)
        self.pt = read_tag(filename,"PT")
        if(self.pt ==-1):
            print("No processing times specified. The program cannot continue.")
            sys.exit()
        else:
            if(len(self.pt) != self.jobs ):
                print("Number of processing times does not match the number of jobs (JOBS={}, length of PT={}). The program cannot continue".format(self.jobs, len(self.pt)) )
                sys.exit()                
            else:
                print_tag_vector("PT", self.pt)
        
        # weights (if not, default weights)
        self.w = read_tag(filename, "W")
        if(self.w ==-1):
            self.w = [1.0 for i in range(self.jobs)]
            print("No weights specified for the jobs. All weights set to 1.0.")  
        else:
            print_tag_vector("W", self.w)

        # due dates (if not,  -1 is assumed)
        self.dd = read_tag(filename, "DD")
        if(self.dd ==-1):
            print("No due dates specified for the jobs. All due dates assummed to be infinite.")           
        else:
            print_tag_vector("DD", self.dd)

        # release dates (if not, 0 is assumed)
        self.r = read_tag(filename,"R")
        if(self.r==-1):
            self.r = [0 for i in range(self.jobs)]
            print("No release dates specified for the jobs. All release dates set to zero.")   
        else:
            print_tag_vector("R", self.r)
        
        print("----- end of SingleMachine instance data from file " + filename + " -------")
        

    # implementation of the computation of the completion times
    def ct(self,sequence):
        completion_time = []
        completion_time.append(self.r[sequence[0]] + self.pt[sequence[0]])
        for i in range(1,len(sequence)):
            completion_time.append(max(completion_time[i-1],self.r[sequence[i]]) + self.pt[sequence[i]])
        return completion_time
   
    

# class to implement the flowshop layout
class FlowShop(Instance):
 
    def __init__(self, filename):

        # initializing additional data (not basic)
        self.machines = 0

        # starting reading
        print("----- Reading FlowShop instance data from file " + filename + " -------")
        # jobs (mandatory data)
        self.jobs = read_tag(filename,"JOBS")
        # if jobs = -1 the program cannot continue
        if(self.jobs ==-1):
            print("No jobs specified. The program cannot continue.")
            sys.exit()
        else:
            print_tag_value("JOBS", self.jobs)
        # machines (another mandatory data)
        self.machines = read_tag(filename, "MACHINES")
        if(self.machines ==-1):
            print("No machines specified. The program cannot continue.")
            sys.exit()
        else:
            print_tag_value("MACHINES", self.machines)

        # processing times (mandatory data, machines in rows, jobs in cols)
        self.pt = read_tag(filename,"PT")
        if(self.pt ==-1):
            print("No processing times specified. The program cannot continue.")
            sys.exit()
        else:
            if(len(self.pt) != self.machines ):
                print("Number of processing times does not match the number of machines (MACHINES={}, length of PT={}). The program cannot continue".format(self.machines, len(self.pt)) )
                sys.exit()
            else:
                for i in range(self.machines):
                    if(len(self.pt[i])!= self.jobs):
                        print("Number of processing times does not match the number of jobs for machine {} (JOBS={}, length of col={}). The program cannot continue".format(i, self.jobs, len(self.pt[i])) )
                        sys.exit()
                print_tag_matrix("PT", self.pt)           
        
        # weights (if not, default weights)
        self.w = read_tag(filename, "W")
        if(self.w ==-1):
            self.w = [1.0 for i in range(self.jobs)]
            print("No weights specified for the jobs. All weights set to 1.0.") 
        else:
            print_tag_vector("W", self.w)

        # due dates (if not,  -1 is assumed)
        self.dd = read_tag(filename, "DD")
        if(self.dd ==-1):
            print("No due dates specified for the jobs. All due dates assummed to be infinite. No due-date related objectives can be computed.")           
        else:
            print_tag_vector("DD", self.dd)

        # release dates (if not, 0 is assumed)
        self.r = read_tag(filename,"R")
        if(self.r==-1):
            self.r = [0 for i in range(self.jobs)]
            print("No release dates specified for the jobs. All release dates set to zero.")    
        else:
            print_tag_vector("R", self.r)

        print("----- end of FlowShop instance data from file " + filename + " -------")    


    # implementation of completion times for FlowShop
    def ct(self, sequence):
        # initializing the completion times
        completion_time = [[0 for j in range(len(sequence))] for i in range(self.machines)]
        # first job in first machine
        completion_time[0][0] = self.r[sequence[0]] + self.pt[0][sequence[0]] 
        # first job in all machines
        for i in range(1,self.machines):
            completion_time[i][0] = completion_time[i-1][0] + self.pt[i][sequence[0]]
        # rest of jobs in first machine
        for j in range(1, len(sequence)):
            completion_time[0][j] =max(completion_time[0][j-1], self.r[sequence[j]]) + self.pt[0][sequence[j]]
        # rest of jobs in rest of machines
        for i in range(1, self.machines):
            for j in range(1, len(sequence)):
                completion_time[i][j] = max(completion_time[i-1][j], completion_time[i][j-1]) + self.pt[i][sequence[j]]
        # computing completion times of each job
        ct = [completion_time[self.machines-1][j] for j in range(len(sequence)) ]
        return ct


# identical parallel machines
class ParallelMachines(Instance):
   def __init__(self, filename):

        # initializing additional data (not basic)
        self.machines = 0

        # starting reading
        print("----- Reading ParallelMachines instance data from file " + filename + " -------")
        # jobs (mandatory data)
        self.jobs = read_tag(filename,"JOBS")
        # if jobs = -1 the program cannot continue
        if(self.jobs ==-1):
            print("No jobs specified. The program cannot continue.")
            sys.exit()
        else:
            print_tag_value("JOBS", self.jobs)
        # machines (another mandatory data)
        self.machines = read_tag(filename, "MACHINES")
        if(self.machines ==-1):
            print("No machines specified. The program cannot continue.")
            sys.exit()
        else:
            print_tag_value("MACHINES", self.machines)

        # processing times (a vector, one pt for jobs). Mandatory
        self.pt = read_tag(filename,"PT")
        if(self.pt ==-1):
            print("No processing times specified. The program cannot continue.")
            sys.exit()
        else:
            if(len(self.pt) != self.jobs ):
                print("Number of processing times does not match the number of jobs (JOBS={}, length of PT={}). The program cannot continue".format(self.jobs, len(self.pt)) )
                sys.exit()                
            else:
                print_tag_vector("PT", self.pt)

        # weights (if not, default weights)
        self.w = read_tag(filename, "W")
        if(self.w ==-1):
            self.w = [1.0 for i in range(self.jobs)]
            print("No weights specified for the jobs. All weights set to 1.0.") 
        else:
            print_tag_vector("W", self.w)

        # due dates (if not,  -1 is assumed)
        self.dd = read_tag(filename, "DD")
        if(self.dd ==-1):
            print("No due dates specified for the jobs. All due dates assummed to be infinite. No due-date related objectives can be computed.")           
        else:
            print_tag_vector("DD", self.dd)

        # release dates (if not, 0 is assumed)
        self.r = read_tag(filename,"R")
        if(self.r==-1):
            self.r = [0 for i in range(self.jobs)]
            print("No release dates specified for the jobs. All release dates set to zero.")    
        else:
            print_tag_vector("R", self.r)

        print("----- end of ParallelMachines instance data from file " + filename + " -------")    
  

    # implementation of completion times for parallel machines (ECT rule)
    # ties are broken with the lowest index
   def ct(self, sequence):

        # initializing completion times in the machines to zero
        ct_machines = [0 for i in range(self.machines)]
        completion_times = [0 for j in range(len(sequence))]

        # assign all jobs
        for j in range(len(sequence)):
            # assign the job to the machine finishing first
            index_machine = find_index_min(ct_machines)
            # increases the completion time of the corresponding machine (and sets the completion time of the job)
            ct_machines[index_machine] = max(ct_machines[index_machine], self.r[sequence[j]]) + self.pt[sequence[j]]
            completion_times[sequence[j]] = ct_machines[index_machine]
    
        return completion_times


class UnrelatedMachines(Instance):
   def __init__(self, filename):

        # initializing additional data (not basic)
        self.machines = 0

        # starting reading
        print("----- Reading UnrelatedMachines instance data from file " + filename + " -------")
        # jobs (mandatory data)
        self.jobs = read_tag(filename,"JOBS")
        # if jobs = -1 the program cannot continue
        if(self.jobs ==-1):
            print("No jobs specified. The program cannot continue.")
            sys.exit()
        else:
            print_tag_value("JOBS", self.jobs)
        # machines (another mandatory data)
        self.machines = read_tag(filename, "MACHINES")
        if(self.machines ==-1):
            print("No machines specified. The program cannot continue.")
            sys.exit()
        else:
            print_tag_value("MACHINES", self.machines)

        # processing times (mandatory data, machines in rows, jobs in cols)
        self.pt = read_tag(filename,"PT")
        if(self.pt ==-1):
            print("No processing times specified. The program cannot continue.")
            sys.exit()
        else:
            if(len(self.pt) != self.machines ):
                print("Number of processing times does not match the number of machines (MACHINES={}, length of PT={}). The program cannot continue".format(self.machines, len(self.pt)) )
                sys.exit()
            else:
                for i in range(self.machines):
                    if(len(self.pt[i])!= self.jobs):
                        print("Number of processing times does not match the number of jobs for machine {} (JOBS={}, length of col={}). The program cannot continue".format(i, self.jobs, len(self.pt[i])) )
                        sys.exit()
                print_tag_matrix("PT", self.pt)     

        # weights (if not, default weights)
        self.w = read_tag(filename, "W")
        if(self.w ==-1):
            self.w = [1.0 for i in range(self.jobs)]
            print("No weights specified for the jobs. All weights set to 1.0.") 
        else:
            print_tag_vector("W", self.w)

        # due dates (if not,  -1 is assumed)
        self.dd = read_tag(filename, "DD")
        if(self.dd ==-1):
            print("No due dates specified for the jobs. All due dates assummed to be infinite. No due-date related objectives can be computed.")           
        else:
            print_tag_vector("DD", self.dd)

        # release dates (if not, 0 is assumed)
        self.r = read_tag(filename,"R")
        if(self.r==-1):
            self.r = [0 for i in range(self.jobs)]
            print("No release dates specified for the jobs. All release dates set to zero.")    
        else:
            print_tag_vector("R", self.r)

        print("----- end of UnrelatedMachines instance data from file " + filename + " -------")    
  

    # implementation of completion times for unrelated parallel machines (ECT rule)
    # ties are broken with the lowest index
   def ct(self, sequence):

        # initializing completion times in the machines to zero
        ct_machines = [0 for i in range(self.machines)]
        completion_times = [0 for j in range(len(sequence))]

        # assign all jobs
        for j in range(len(sequence)):
            # construct what completion times would be if the job is assigned to each machine
            next_ct = [max(ct_machines[i],self.r[sequence[j]]) + self.pt[i][sequence[j]] for i in range(self.machines)]
            # assign the job to the machine finishing first
            index_machine = find_index_min(next_ct)
            # increases the completion time of the corresponding machine (and sets the completion time of the job)
            ct_machines[index_machine] = max(ct_machines[index_machine], self.r[sequence[j]]) + self.pt[index_machine][sequence[j]]
            completion_times[sequence[j]] = ct_machines[index_machine]
    
        return completion_times

