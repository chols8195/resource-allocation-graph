import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

class ResourceAllocationGraph:
    def __init__(self, numProcesses, numResources, statements = None):
        # initialize the instance variables 
        self.step = 0
        self.numberProcesses = numProcesses 
        self.numberResources = numResources
        self.systemDeadlock = False
        
        # tuples to keep track of the processes and resources
        self.statementsList = statements if statements is not None else []
        
        # edge setup for graph 
        self.claimEdge = [] 
        self.requestEdge = []
        
        # representing the matrix
        self.matrixAlloc = np.zeros((numProcesses, numResources), dtype=int)
        self.matrixRequest = np.zeros((numProcesses, numResources), dtype=int)
        self.availableResources = np.ones(numResources, dtype = int)
        
        # keep track of deadlocked processes
        self.deadlockedProcesses = []
        
    def addStatement(self, statement):
        # add to statement list
        self.statementsList.append(statement)
        
    def setupDeadlockScenario(self):
        # deadlock scenario
        self.numberProcesses = 3
        self.numberResources = 3
        self.statementsList = [
            "P0 requests R0",
            "P0 holds R0",
            "P1 requests R1",
            "P1 holds R1",
            "P2 requests R2",
            "P2 holds R2",
            "P0 requests R1",
            "P1 requests R2",
            "P2 requests R0"
        ]
            
        # reset matrices and edges 
        self.matrixAlloc = np.zeros((self.numberProcesses, self.numberResources), dtype=int)
        self.matrixRequest = np.zeros((self.numberProcesses, self.numberResources), dtype=int)
        self.availableResources = np.ones(self.numberResources, dtype=int)
        self.claimEdge = []
        self.requestEdge = []
        self.deadlockedProcesses = []
    
    # simulate the resource allocation graph based on statement list    
    def simulate(self):
        # use matplotlib to create a graph
        plt.rcParams['toolbar'] = 'None' # remove toolbar
        plt.axis('off') # turn off axis
        plt.ion() # turn on interactive mode
        plt.figure(figsize = (10, 6)) # set figure size
        plt.show() # clear the figure
        
        # print initial state of the graph
        print("Initial State:")
        print("Number of Processes:", {self.numberProcesses})
        print("Number of Resources:", {self.numberResources})
        print("Available Resources:", self.availableResources)
        
        # loop through statement list and update graph
        while self.step < len(self.statementsList):
            if self.systemDeadlock: # systemDeadlock = True
                self.step = len(self.statementsList)
                print("System is deadlocked! Stopping program...")
                break
            
            # parse into data structure
            self.parseStatement()
            
            # check for deadlock after each statement
            self.detectDeadlock()
            if self.deadlockedProcesses:
                print("Deadlock detected!")
                print(f"Deadlocked Processes: {', '.join(f'P{p}' for p in self.deadlockedProcesses)}") # prints out deadlocked processes
                if len(self.deadlockedProcesses) == self.numberProcesses:
                    self.systemDeadlock = True
                    print("System is deadlocked! Stopping program...")
            
            self.printState()
            self.drawGraph()
    
    # parse a statement and update data structure        
    def parseStatement(self):
        statement = self.statementsList[self.step]
        print(f"\nStep {self.step + 1}: {statement}")
        
        splitStatement = statement.split(" ")
        
        # parse and split statement to be useable
        processNum = int(splitStatement[0][1]) # this is the process number
        action = splitStatement[1] # this is the action --> request or hold
        resourceNum = int(splitStatement[2][1]) # this is the resource number
        
        # skip process if deadlocked 
        if processNum in self.deadlockedProcesses:
            print(f"P{processNum} is deadlocked. Ignoring statement")
            self.step += 1
            return
        
        # process statement based on action 
        if action == "requests":
            self.matrixRequest[processNum][resourceNum] = 1
            self.requestEdge.append((processNum, resourceNum + self.numberProcesses)) # process to resource edge
        elif action == "holds":
            # resource is held by process
            if self.matrixRequest[processNum][resourceNum] == 1:
                # check if resource is available
                if self.availableResources[resourceNum] == 1:
                    # remove request 
                    self.matrixRequest[processNum][resourceNum] = 0 
                    if (processNum, resourceNum + self.numberProcesses) in self.requestEdge:
                        self.requestEdge.remove((processNum, resourceNum + self.numberProcesses))
                    
                    # update available resources by allocating the resource    
                    self.matrixAlloc[processNum][resourceNum] = 1
                    self.availableResources[resourceNum] = 0
                    self.claimEdge.append((resourceNum + self.numberProcesses, processNum)) # resource to process edge
                    
                    print(f"P{processNum} allocated to R{resourceNum}")
                else:
                    print(f"Resource R{resourceNum} is not available")
        
            else:
                print (f"P{processNum} is not requesting R{resourceNum}")

        elif action == "releases":
            # process releases a resource
            if self.matrixAlloc[processNum][resourceNum] == 1:
                self.matrixAlloc[processNum][resourceNum] = 0
                self.availableResources[resourceNum] = 1
                
                # remove claim edge
                if (resourceNum + self.numberProcesses, processNum) in self.claimEdge:
                    self.claimEdge.remove((resourceNum + self.numberProcesses, processNum))
                
                print(f"P{processNum} released R{resourceNum}")
                
                # check if any process is waiting for a resource
                self.checkWaitingProcesses(resourceNum)
            
            else:
                print(f"P{processNum} is not holding R{resourceNum}")
        
        # increment step 
        self.step += 1
        
    def checkWaitingProcesses(self, resourceNum):
        # check for any process that is waiting for the resource
        for i in range(self.numberProcesses):
            if self.matrixRequest[i][resourceNum] == 1 and i not in self.deadlockedProcesses:
                #
                self.matrixRequest[i][resourceNum] = 0
                if (i, resourceNum + self.numberProcesses) in self.requestEdge:
                    self.requestEdge.remove((i, resourceNum + self.numberProcesses))
                    
                self.matrixAlloc[i][resourceNum] = 1
                self.availableResources[resourceNum] = 0
                self.claimEdge.append((resourceNum + self.numberProcesses, i)) # resource to process edge
                
                print(f"P{i} allocated to R{resourceNum}")
                return # one process is allocated the resource
    
    def detectDeadlock(self):
        # detect deadlock using cycle detection algorithm
        graph = nx.DiGraph()
        
        # add edges to the graph
        processes = list(range(self.numberProcesses))
        resources = list(range(self.numberProcesses, self.numberProcesses + self.numberResources))
        graph.add_nodes_from(processes + resources) # add process and resource nodes
        graph.add_edges_from(self.claimEdge + self.requestEdge)
        
        # check for cycle in the graph
        self.deadlockedProcesses = []
        for cycle in nx.simple_cycles(graph):
            for node in cycle:
                if node < self.numberProcesses and node not in self.deadlockedProcesses:
                    self.deadlockedProcesses.append(node) # add process to deadlocked processes
        
        return len(self.deadlockedProcesses) > 0 # return true if deadlock is detected
    
    def printState(self):
        print("\nCurrent State:")
        print("Allocation Matrix:")
        print(self.matrixAlloc)
        print("Request Matrix:")
        print(self.matrixRequest)
        print("Available Resources (available = 1, allocated = 0):")
        print(self.availableResources)
        print("\n")
    
    def drawGraph(self):
        plt.clf() # clear the figure
        graph = nx.DiGraph()
        
        # create nodes for processes and resources
        processes = list(range(self.numberProcesses))
        resources = list(range(self.numberProcesses, self.numberProcesses + self.numberResources))
        
        # create labels
        labels = {}
        for i in processes:
            labels[i] = f"P{i}"
        for j in resources:
            labels[j] = f"R{j - self.numberProcesses}"
            
        # add nodes and edges to the graph
        graph.add_nodes_from(processes + resources) # add process and resource nodes
        graph.add_edges_from(self.claimEdge + self.requestEdge)
        
        # set positions of nodes and draw graph
        pos = nx.bipartite_layout(graph, nodes = processes, align = 'horizontal')
        
        # Draw process nodes as circles
        nx.draw_networkx_nodes(graph, pos, nodelist = processes, 
                               node_color = ['red' if i in self.deadlockedProcesses else 'blue' for i in processes], 
                               node_size = 600, alpha = 1, node_shape = 'o') # process nodes as circles
        
        # Draw resource nodes as rectangles (squares)
        nx.draw_networkx_nodes(graph, pos, nodelist = resources, node_color = 'green', 
                               node_size = 700, alpha = 1, node_shape = 's') # resource nodes as squares
        
        # Draw edges with different colors for request and claim
        nx.draw_networkx_edges(graph, pos, edgelist = self.requestEdge, width = 1, alpha = 1, 
                               arrows = True, arrowstyle = '->', arrowsize = 20, edge_color = 'red')
        nx.draw_networkx_edges(graph, pos, edgelist = self.claimEdge, width = 1, alpha = 1, arrows = True,
                               arrowstyle = '->', arrowsize = 20, edge_color = 'blue')
        
        # draw labels
        nx.draw_networkx_labels(graph, pos, labels, font_size = 12, font_color = 'white')
        
        # set title and show graph
        if self.deadlockedProcesses:
            plt.title(f"Single Instance Resource Allocation Graph (Deadlock Detected):\n {', '.join(f'P{i}' for i in self.deadlockedProcesses)}", fontsize = 12)
        else:
            plt.title("Single Instance Resource Allocation Graph", fontsize = 12)
        
        # make sure axis is off
        plt.axis('off')
        plt.tight_layout()
        plt.pause(2) # pause for 2 seconds to show graph
        plt.savefig(f"single_instance{self.step}.png")
        
        if self.step == len(self.statementsList):
            self.shutdown()
    
    # prompt user to shutdown the program
    def shutdown(self):
        input("Press enter to exit")

if __name__ == '__main__':
    # create a manager
    rm = ResourceAllocationGraph(3, 3)
    
    # Setup deadlock scenario directly
    rm.setupDeadlockScenario()
    
    # simulate the resource allocation graph
    rm.simulate()