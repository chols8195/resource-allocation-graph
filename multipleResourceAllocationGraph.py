import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

class MultipleInstanceResourceManager:
    # resource allocation manager for multiple instance resources
    # uses resource allocation graph for deadlock detection
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
        
        # Keep track of resource allocation
        # Not using banker's algorithm, just resource allocation graph
        # Still maintain this for informational purposes
        self.maxNeeds = np.zeros((numProcesses, numResources), dtype=int)
        
        # keep track of deadlocked processes
        self.deadlockedProcesses = []

    def addStatement(self, statement):
        # add a statement to the statement list
        self.statementsList.append(statement)
        
    def scenario(self, scenarioType):
        # set up a predefined scenario
        if scenarioType == "deadlock":
            # multiple instance with deadlock
            self.numberProcesses = 3
            self.numberResources = 2
            self.resourceInstance = [2, 2]
            self.statementsList = [
                "p0 requests r0",
                "p0 holds r0",
                "p1 requests r0",
                "p1 holds r0",
                "p0 requests r1",
                "p0 holds r1",
                "p2 requests r1",
                "p2 holds r1",
                "p1 requests r1",
                "p2 requests r0"
            ]
            
        elif scenarioType == "nodeadlock":
            # multiple instance with no deadlock
            self.numberProcesses = 3
            self.numberResources = 2
            self.resourceInstance = [2, 2]
            self.statementsList = [
                "p0 requests r0",
                "p0 holds r0",
                "p1 requests r0",
                "p1 holds r0",
                "p0 requests r1",
                "p0 holds r1",
                "p0 releases r0",
                "p2 requests r0",
                "p2 holds r0",
                "p1 releases r0",
                "p2 requests r1",
                "p2 holds r1",
                "p1 requests r0",
                "p1 holds r0"
            ]
        
        # reset matrices and edges
        self.matrixAlloc = np.zeros((self.numberProcesses, self.numberResources), dtype=int)
        self.requestMatrix = np.zeros((self.numberProcesses, self.numberResources), dtype=int)
        self.availableResources = np.array(self.resourceInstance.copy(), dtype=int)
        self.maxNeeds = np.zeros((self.numberProcesses, self.numberResources), dtype=int)
        self.requestEdge = []
        self.claimEdge = []
        self.deadlockedProcesses = []

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
                self.shutdownPrompt()
                break

            # parse statement into data structure
            self.parseStatement()
            
            # check for deadlock after each statement
            if self.detect_deadlock():
                print("Deadlock detected!")
                print(f"Deadlocked processes: {', '.join(f'p{i}' for i in self.deadlockedProcesses)}")
                if len(self.deadlockedProcesses) == self.numberProcesses:
                    self.system_deadlocked = True
                    print("System completely deadlocked, halting program")
            
            # print matrix states
            self.print_matrixState()
            
            # show graph for each statement
            self.draw_graph()

    def parseStatement(self):
        # parse a statement and update matrices accordingly
        statement = self.statementsList[self.step]
        print(f"\nStep {self.step + 1}: {statement}")
        
        splitStatement = statement.split(" ")
        
        # split out statement into a usable format
        processNum = int(splitStatement[0][1])
        action = splitStatement[1]
        resourceNum = int(splitStatement[2][1])
        
        # skip processing if the process is deadlocked
        if processNum in self.deadlockedProcesses:
            print(f"p{processNum} is deadlocked, ignoring statement '{statement}'")
            self.step += 1
            return
        
        # process the statement based on action
        if action == "requests":
            # mark the request in the request matrix
            self.requestMatrix[processNum][resourceNum] += 1
            
            # Just track maximum resources needed for informational purposes
            # Not needed for resource allocation graph deadlock detection
            self.maxNeeds[processNum][resourceNum] = max(
                self.maxNeeds[processNum][resourceNum],
                self.matrixAlloc[processNum][resourceNum] + self.requestMatrix[processNum][resourceNum]
            )
            
            # for graph visualization, add request edge
            # for multiple instances we'll only add one request edge per resource type
            # (even if requesting multiple instances)
            if (processNum, resourceNum + self.numberProcesses) not in self.requestEdge:
                self.requestEdge.append((processNum, resourceNum + self.numberProcesses))
            
        elif action == "holds":
            # resource is holds to the process
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
                edgeToRemove = (resourceNum + self.numberProcesses, processNum)
                if edgeToRemove in self.claimEdge:
                    self.claimEdge.remove(edgeToRemove)
                
                print(f"p{processNum} released r{resourceNum}, now holds {self.matrixAlloc[processNum][resourceNum]} instances")
                
                # check if any process is waiting for this resource
                self.checkPendingRequests(resourceNum)
            else:
                print(f"Error: p{processNum} doesn't hold any instances of r{resourceNum}")
        
        # increment step counter
        self.step += 1

    def checkPendingRequests(self, resourceNum):
        # check if any process is waiting for the released resource
        if self.availableResources[resourceNum] <= 0:
            return
            
        # find processes that requested this resource
        for i in range(self.numberProcesses):
            if self.requestMatrix[i][resourceNum] > 0 and i not in self.deadlockedProcesses:
                # grant the resource to this process
                self.requestMatrix[i][resourceNum] -= 1
                self.matrixAlloc[i][resourceNum] += 1
                self.availableResources[resourceNum] -= 1
                
                # add ownership edge for visualization
                self.claimEdge.append((resourceNum + self.numberProcesses, i))  # Fixed: 'p' to 'i'
                
                # if no more requests, remove request edge
                if self.requestMatrix[i][resourceNum] == 0:
                    if (i, resourceNum + self.numberProcesses) in self.requestEdge:
                        self.requestEdge.remove((i, resourceNum + self.numberProcesses))
                
                print(f"p{i} was granted r{resourceNum} from waiting queue, now holds {self.matrixAlloc[i][resourceNum]} instances")  # Fixed: "holds" to "granted"
                
                # only grant to one process (this can be modified if needed)
                return

    def detect_deadlock(self):
        # Resource Allocation Graph (RAG) method for deadlock detection
        # Create a directed graph
        graph = nx.DiGraph()
        
        # Add nodes for processes and resources
        processes = list(range(self.numberProcesses))
        resources = list(range(self.numberProcesses, self.numberProcesses + self.numberResources))
        graph.add_nodes_from(processes + resources)
        
        # Add all edges (request edges and claim edges)
        graph.add_edges_from(self.requestEdge + self.claimEdge)
        
        # Check for cycles in the graph
        # A cycle in a resource allocation graph indicates a deadlock
        self.deadlockedProcesses = []
        for cycle in nx.simple_cycles(graph):
            for node in cycle:
                # Only add process nodes to deadlocked processes list
                if node < self.numberProcesses and node not in self.deadlockedProcesses:
                    self.deadlockedProcesses.append(node)
                    
        # Return True if there are deadlocked processes
        return len(self.deadlockedProcesses) > 0

    def print_matrixState(self):
        # print the current state of matrices
        print("\nCurrent State:")
        print("Allocation Matrix:")
        print(self.matrixAlloc)
        print("\nRequest Matrix:")
        print(self.requestMatrix)
        print("\nMax Needs Matrix:")
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
            labels[i] = f'P{i}'
        for j in resources:
            resource_id = j - self.numberProcesses
            allocated = sum(self.matrixAlloc[:, resource_id])
            total = self.resourceInstance[resource_id]
            labels[j] = f'r{resource_id}\n({allocated}/{total})'
        
        # add nodes and edges to the graph
        graph.add_nodes_from(processes + resources)
        graph.add_edges_from(self.requestEdge + self.claimEdge)
        
        # set fixed positions for nodes to prevent movement
        # create a fixed layout with processes on left, resources on right
        pos = nx.bipartite_layout(graph, nodes=processes + resources, align = 'horizontal')
        processSpacing = 1.0 / (self.numberProcesses + 1)
        resourceSpacing = 1.0 / (self.numberResources + 1)
        
        # position processes on the left side
        for i, p in enumerate(processes):
            pos[i] = (0.25, (i + 1) * processSpacing)
            
        # position resources on the right side
        for i, j in enumerate(resources):
            pos[j] = (0.75, (i + 1) * resourceSpacing)
        
        # draw process nodes as circles
        nx.draw_networkx_nodes(graph, 
                               pos,
                               nodelist=processes,
                               node_color=['red' if p in self.deadlockedProcesses else 'blue' for p in processes],  # Fixed: 'i' to 'p'
                               node_size=600,
                               alpha=1,
                               node_shape='o')
        
        # draw resource nodes as squares
        nx.draw_networkx_nodes(graph, 
                               pos,
                               nodelist = resources,
                               node_color = 'green',
                               node_size = 700,
                               alpha = 1,
                               node_shape = 's')
        
        # draw request edges with straight lines
        nx.draw_networkx_edges(graph, pos,
                              edgelist = self.requestEdge,
                              width=1, 
                              alpha=1, 
                              arrows=True, 
                              arrowstyle = '->', 
                              arrowsize=20,
                              edge_color='red')
        
        # draw ownership edges with straight lines
        nx.draw_networkx_edges(graph, pos,
                              edgelist = self.claimEdge,
                              width = 1, 
                              alpha = 1, 
                              arrows = True, 
                              arrowstyle = '->', 
                              arrowsize = 20,
                              edge_color = 'blue')
        
        # draw labels
        nx.draw_networkx_labels(graph, pos, labels, font_size=12, font_color='white')
        
        # set title based on deadlock status
        if self.deadlockedProcesses:
            plt.title(f"Multiple Instance Resource Allocation Graph\nDeadlock Detected!", 
                     color='red', fontsize=12)
        else:
            plt.title(f"Multiple Instance Resource Allocation Graph", fontsize=12)
        
        # make sure axis stays gone
        plt.axis("off")
        plt.tight_layout()
        plt.pause(2)
        
        # if this is the final step, prompt for shutdown
        if self.step == len(self.statementsList):
            self.shutdownPrompt()

    def shutdownPrompt(self):
        # prompt for program shutdown
        input("Press enter to exit...")

# example usage
if __name__ == '__main__':
    # create a manager
    rm = MultipleInstanceResourceManager(3, 2, [2, 2])
    
    # choose a scenario
    scenario = input("Enter scenario (1 = deadlock, 2 = nodeadlock): ")  # Fixed: "noDeadlock" to "nodeadlock"
    if scenario == '1':
        rm.scenario("deadlock")
    elif scenario == '2':
        rm.scenario("nodeadlock")  # Fixed: Match scenario name with input prompt
    else:
        print("Invalid scenario. Defaulting to deadlock scenario.")
        rm.scenario("deadlock")
    
    # run the simulation
    rm.simulate()