import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

class MultipleInstanceResourceManager:
    # resource allocation manager for multiple instance resources
    # uses the banker's algorithm for deadlock detection
    def __init__(self, numProcesses, numResources, resourceInstance, statements = None):
        # initialize the instance variables
        self.step = 0
        self.numberProcesses = numProcesses
        self.numberResources = numResources
        self.system_deadlocked = False
        
        # set up resource instances
        self.resourceInstance = resourceInstance
        
        # statements to process
        self.statementsList = statements if statements else []
        
        # edge setup for graph
        # for multiple instances, we create one edge per allocated resource
        self.requestEdge = []  # (process_id, resource_id + numProcesses)
        self.claimEdge = []    # (resource_id + numProcesses, process_id)
        
        # representing the matrix
        # allocation matrix: how many resources of each type are allocated to each process
        self.matrixAlloc = np.zeros((numProcesses, numResources), dtype=int)
        
        # request matrix: how many resources of each type are requested by each process
        self.requestMatrix = np.zeros((numProcesses, numResources), dtype=int)
        
        # available resources: how many instances of each resource type are available
        self.availableResources = np.array(self.resourceInstance.copy(), dtype=int)
        
        # max needs of each process (for banker's algorithm)
        # will be updated as we process statements
        self.maxNeeds = np.zeros((numProcesses, numResources), dtype=int)
        
        # keep track of deadlocked processes
        self.deadlocked_processes = []

    def add_statement(self, statement):
        # add a statement to the statement list
        self.statementsList.append(statement)
        
    def set_predefined_scenario(self, scenario_type):
        # set up a predefined scenario
        if scenario_type == "deadlock":
            # multiple instance with deadlock
            self.numberProcesses = 3
            self.numberResources = 2
            self.resourceInstance = [2, 2]
            self.statementsList = [
                "p0 requests r0",
                "p0 granted r0",
                "p1 requests r0",
                "p1 granted r0",
                "p0 requests r1",
                "p0 granted r1",
                "p2 requests r1",
                "p2 granted r1",
                "p1 requests r1",
                "p2 requests r0"
            ]
            
        elif scenario_type == "nodeadlock":
            # multiple instance with no deadlock
            self.numberProcesses = 3
            self.numberResources = 2
            self.resourceInstance = [2, 2]
            self.statementsList = [
                "p0 requests r0",
                "p0 granted r0",
                "p1 requests r0",
                "p1 granted r0",
                "p0 requests r1",
                "p0 granted r1",
                "p0 releases r0",
                "p2 requests r0",
                "p2 granted r0",
                "p1 releases r0",
                "p2 requests r1",
                "p2 granted r1"
            ]
        
        # reset matrices and edges
        self.matrixAlloc = np.zeros((self.numberProcesses, self.numberResources), dtype=int)
        self.requestMatrix = np.zeros((self.numberProcesses, self.numberResources), dtype=int)
        self.availableResources = np.array(self.resourceInstance.copy(), dtype=int)
        self.maxNeeds = np.zeros((self.numberProcesses, self.numberResources), dtype=int)
        self.requestEdge = []
        self.claimEdge = []
        self.deadlocked_processes = []

    def simulate(self):
        # run the simulation based on the statement list
        # use matplotlib to create a graph
        plt.rcParams['toolbar'] = 'None'  # remove toolbar
        plt.axis('off')  # turn off axis
        plt.ion()  # turn on interactive mode
        plt.figure(figsize=(10, 6))  # set figure size
        plt.show()  # clear the figure
        
        # print initial state
        print(f"Initial State:")
        print(f"Number of processes: {self.numberProcesses}")
        print(f"Number of resources: {self.numberResources}")
        print(f"Resource instances: {self.resourceInstance}")
        print(f"Available resources: {self.availableResources}")
        print("\nStarting simulation...\n")
        
        while self.step < len(self.statementsList):
            if self.system_deadlocked:
                # if system is deadlocked then halt more drawing and parsing
                self.step = len(self.statementsList)
                self.shutdown_prompt()
                break

            # parse statement into data structure
            self.parse_statement()
            
            # check for deadlock after each statement
            if self.detect_deadlock():
                print("Deadlock detected!")
                print(f"Deadlocked processes: {', '.join(f'p{i}' for i in self.deadlocked_processes)}")
                if len(self.deadlocked_processes) == self.numberProcesses:
                    self.system_deadlocked = True
                    print("System completely deadlocked, halting program")
            
            # print matrix states
            self.print_matrix_state()
            
            # show graph for each statement
            self.draw_graph()

    def parse_statement(self):
        # parse a statement and update matrices accordingly
        statement = self.statementsList[self.step]
        print(f"\nStep {self.step + 1}: {statement}")
        
        split_statement = statement.split(" ")
        
        # split out statement into a usable format
        processNum = int(split_statement[0][1])
        action = split_statement[1]
        resourceNum = int(split_statement[2][1])
        
        # skip processing if the process is deadlocked
        if processNum in self.deadlocked_processes:
            print(f"p{processNum} is deadlocked, ignoring statement '{statement}'")
            self.step += 1
            return
        
        # process the statement based on action
        if action == "requests":
            # mark the request in the request matrix
            self.requestMatrix[processNum][resourceNum] += 1
            
            # update.maxNeeds for banker's algorithm
            self.maxNeeds[processNum][resourceNum] = max(
                self.maxNeeds[processNum][resourceNum],
                self.matrixAlloc[processNum][resourceNum] + self.requestMatrix[processNum][resourceNum]
            )
            
            # for graph visualization, add request edge
            # for multiple instances we'll only add one request edge per resource type
            # (even if requesting multiple instances)
            if (processNum, resourceNum + self.numberProcesses) not in self.requestEdge:
                self.requestEdge.append((processNum, resourceNum + self.numberProcesses))
            
        elif action == "granted":
            # resource is granted to the process
            if self.requestMatrix[processNum][resourceNum] > 0:
                # check if resource is available
                if self.availableResources[resourceNum] > 0:
                    # reduce the request
                    self.requestMatrix[processNum][resourceNum] -= 1
                    
                    # add to allocation
                    self.matrixAlloc[processNum][resourceNum] += 1
                    
                    # reduce available resources
                    self.availableResources[resourceNum] -= 1
                    
                    # add ownership edge for visualization
                    self.claimEdge.append((resourceNum + self.numberProcesses, processNum))
                    
                    # if no more requests, remove request edge
                    if self.requestMatrix[processNum][resourceNum] == 0:
                        if (processNum, resourceNum + self.numberProcesses) in self.requestEdge:
                            self.requestEdge.remove((processNum, resourceNum + self.numberProcesses))
                    
                    print(f"p{processNum} now holds {self.matrixAlloc[processNum][resourceNum]} instances of r{resourceNum}")
                else:
                    print(f"Error: No available instances of r{resourceNum}")
            else:
                print(f"Error: p{processNum} didn't request r{resourceNum}")
        
        elif action == "releases":
            # process releases a resource
            if self.matrixAlloc[processNum][resourceNum] > 0:
                # reduce allocation
                self.matrixAlloc[processNum][resourceNum] -= 1
                
                # increase available resources
                self.availableResources[resourceNum] += 1
                
                # for graph visualization, remove one ownership edge
                edge_to_remove = (resourceNum + self.numberProcesses, processNum)
                if edge_to_remove in self.claimEdge:
                    self.claimEdge.remove(edge_to_remove)
                
                print(f"p{processNum} released r{resourceNum}, now holds {self.matrixAlloc[processNum][resourceNum]} instances")
                
                # check if any process is waiting for this resource
                self.check_pending_requests(resourceNum)
            else:
                print(f"Error: p{processNum} doesn't hold any instances of r{resourceNum}")
        
        # increment step counter
        self.step += 1

    def check_pending_requests(self, resourceNum):
        # check if any process is waiting for the released resource
        if self.availableResources[resourceNum] <= 0:
            return
            
        # find processes that requested this resource
        for i in range(self.numberProcesses):
            if self.requestMatrix[i][resourceNum] > 0 and i not in self.deadlocked_processes:
                # grant the resource to this process
                self.requestMatrix[i][resourceNum] -= 1
                self.matrixAlloc[i][resourceNum] += 1
                self.availableResources[resourceNum] -= 1
                
                # add ownership edge for visualization
                self.claimEdge.append((resourceNum + self.numberProcesses, p))
                
                # if no more requests, remove request edge
                if self.requestMatrix[i][resourceNum] == 0:
                    if (i, resourceNum + self.numberProcesses) in self.requestEdge:
                        self.requestEdge.remove((i, resourceNum + self.numberProcesses))
                
                print(f"p{i} was granted r{resourceNum} from waiting queue, now holds {self.matrixAlloc[i][resourceNum]} instances")
                
                # only grant to one process (this can be modified if needed)
                return

    def detect_deadlock(self):
        # detect deadlock using banker's algorithm
        # calculate need matrix (max - allocation)
        need_matrix = self.maxNeeds - self.matrixAlloc
        
        # try to find a safe sequence
        work = self.availableResources.copy()
        finish = [False] * self.numberProcesses
        safe_sequence = []
        
        # keep track of deadlocked processes
        self.deadlocked_processes = []
        
        # continue until no change or all processes are finished
        while True:
            found = False
            for i in range(self.numberProcesses):
                if not finish[i]:
                    # check if process p's needs can be satisfied
                    can_allocate = True
                    for j in range(self.numberResources):
                        if need_matrix[i][j] > work[j]:
                            can_allocate = False
                            break
                    
                    if can_allocate:
                        # process p can finish
                        for j in range(self.numberResources):
                            work[j] += self.matrixAlloc[i][j]
                        finish[i] = True
                        safe_sequence.append(i)
                        found = True
                        break
            
            if not found:
                break
        
        # if any process couldn't finish, there's a deadlock
        for i in range(self.numberProcesses):
            if not finish[i]:
                self.deadlocked_processes.append(i)
        
        if self.deadlocked_processes:
            print("banker's algorithm found potential deadlock")
            print(f"no safe sequence exists for processes: {', '.join(f'p{i}' for i in self.deadlocked_processes)}")
        else:
            print(f"system is in a safe state. safe sequence: {' -> '.join(f'p{i}' for i in safe_sequence)}")
        
        return len(self.deadlocked_processes) > 0

    def print_matrix_state(self):
        # print the current state of matrices
        print("\nCurrent System State:")
        print("Allocation Matrix (how many resources of each type are allocated to each process):")
        print(self.matrixAlloc)
        print("\nRequest Matrix (how many resources of each type are requested by each process):")
        print(self.requestMatrix)
        print("\nMax Needs Matrix (maximum number of each resource type needed by each process):")
        print(self.maxNeeds)
        print("\nAvailable Resources:")
        print(self.availableResources)
        print("\n")

    def draw_graph(self):
        # draw the resource allocation graph using networkx
        plt.clf()
        graph = nx.DiGraph()
        
        # create nodes for processes and resources
        processes = list(range(self.numberProcesses))
        resources = list(range(self.numberProcesses, self.numberProcesses + self.numberResources))
        
        # create labels
        labels = {}
        for i in processes:
            labels[i] = f'p{i}'
        for j in resources:
            resource_id = j - self.numberProcesses
            allocated = sum(self.matrixAlloc[:, resource_id])
            total = self.resourceInstance[resource_id]
            labels[j] = f'r{resource_id}\n({allocated}/{total})'
        
        # add nodes
        graph.add_nodes_from(processes + resources)
        
        # add edges
        graph.add_edges_from(self.requestEdge + self.claimEdge)
        
        # set fixed positions for nodes to prevent movement
        # create a fixed layout with processes on left, resources on right
        pos = {}
        process_spacing = 1.0 / (self.numberProcesses + 1)
        resource_spacing = 1.0 / (self.numberResources + 1)
        
        # position processes on the left side
        for i, p in enumerate(processes):
            pos[i] = (0.25, (i + 1) * process_spacing)
            
        # position resources on the right side
        for i, j in enumerate(resources):
            pos[j] = (0.75, (i + 1) * resource_spacing)
        
        # draw process nodes as circles
        nx.draw_networkx_nodes(graph, pos,
                              nodelist=processes,
                              node_color=['red' if p in self.deadlocked_processes else 'blue' for i in processes],
                              node_size=600,
                              alpha=1,
                              node_shape='o')
        
        # draw resource nodes as squares
        nx.draw_networkx_nodes(graph, pos,
                              nodelist=resources,
                              node_color='green',
                              node_size=700,
                              alpha=1,
                              node_shape='s')
        
        # draw request edges with straight lines
        nx.draw_networkx_edges(graph, pos,
                              edgelist=self.requestEdge,
                              width=1, alpha=1, arrows=True, 
                              arrowstyle='->', arrowsize=20,
                              edge_color='red')
        
        # draw ownership edges with straight lines
        nx.draw_networkx_edges(graph, pos,
                              edgelist=self.claimEdge,
                              width=1, alpha=1, arrows=True, 
                              arrowstyle='->', arrowsize=20,
                              edge_color='blue')
        
        # draw labels
        nx.draw_networkx_labels(graph, pos, labels, font_size=12, font_color='white')
        
        # set title based on deadlock status
        if self.deadlocked_processes:
            plt.title(f"Multiple Instance Resource Allocation Graph (Deadlocked Detected):\n {', '.join(f'p{i}' for i in self.deadlocked_processes)}", 
                     color='red', fontsize=12)
        else:
            plt.title(f"Multiple Instance Resource Allocation Graph", fontsize=12)
        
        # make sure axis stays gone
        plt.axis("off")
        plt.tight_layout()
        plt.pause(2)
        
        # if this is the final step, prompt for shutdown
        if self.step == len(self.statementsList):
            self.shutdown_prompt()

    def shutdown_prompt(self):
        # prompt for program shutdown
        input("Press enter to exit...")

# example usage
if __name__ == '__main__':
    # create a manager
    rm = MultipleInstanceResourceManager(3, 2, [2, 2])
    
    # choose a scenario
    scenario = input("Choose a scenario (1 = deadlock, 2 = noDeadlock): ")
    if scenario == '1':
        rm.set_predefined_scenario("deadlock")
    elif scenario == '2':
        rm.set_predefined_scenario("noDeadlock")
    else:
        print("Invalid scenario. Defaulting to deadlock scenario.")
        rm.set_predefined_scenario("deadlock")
    
    # run the simulation
    rm.simulate()