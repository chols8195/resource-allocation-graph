import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# print the matrices with labels to know what is the process and what is the resource
def printLabels(matrix, rowPrefix="P", colPrefix="R"):
    # fixed column width to ensure consistent spacing
    col_width = 3
    
    # calculate table dimensions
    num_rows = matrix.shape[0]
    num_cols = matrix.shape[1]
    
    # calculate width of the entire table
    table_width = 7 + (num_cols * (col_width + 1))
    
    # create top border
    print("+" + "-" * (table_width - 2) + "+")
    
    # create header row with resource labels
    header = "|     "
    for j in range(num_cols):
        header += f" {colPrefix}{j} "
    header += "|"
    print(header)
    
    # create separator line
    print("+" + "-" * (table_width - 2) + "+")
    
    # create rows with process labels and values
    for i in range(num_rows):
        row = f"| {rowPrefix}{i}  |"
        for j in range(num_cols):
            row += f" {matrix[i][j]} " + " "
        # remove the last extra space and add the closing border
        row = row[:-1] + "|"
        print(row)
    
    # create bottom border
    print("+" + "-" * (table_width - 2) + "+")
        
class MultipleInstanceResourceManager:
    # resource allocation manager for multiple instance resources
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
        self.requestEdge = []  
        self.claimEdge = []    
        
        # representing the matrix
        self.matrixAlloc = np.zeros((numProcesses, numResources), dtype=int)
        self.requestMatrix = np.zeros((numProcesses, numResources), dtype=int)
        self.availableResources = np.array(self.resourceInstance.copy(), dtype=int)
        
        # keep track of deadlocked processes
        self.deadlockedProcesses = []

    def addStatement(self, statement):
        # add a statement to the statement list
        self.statementsList.append(statement)
        
    def scenario(self, scenarioType):
        # set up a predefined scenario
        
        if scenarioType == "deadlock":
            self.numberProcesses = 3
            self.numberResources = 3
            self.resourceInstance = [2, 2, 2]
            self.statementsList = [
                "P0 requests R0",
                "P0 holds R0",
                "P1 requests R0",
                "P1 holds R0",
                "P2 requests R2",
                "P2 holds R2",
                "P0 requests R1",
                "P0 holds R1",
                "P2 requests R1",
                "P2 holds R1",
                "P1 requests R2",
                "P1 holds R2",
                "P0 requests R2",
                "P2 requests R0",
                "P1 requests R1"
            ]
            
        elif scenarioType == "noDeadlock":
        # multiple instance with no deadlock
            self.numberProcesses = 3
            self.numberResources = 3
            self.resourceInstance = [2, 2, 2]
            self.statementsList = [
                "P0 requests R0",
                "P0 holds R0",
                "P1 requests R0",
                "P1 holds R0",
                "P0 requests R1",
                "P0 holds R1",
                "P0 releases R0",
                "P2 requests R0",
                "P2 holds R0",
                "P1 releases R0",
                "P2 requests R2",
                "P2 holds R2",
                "P1 requests R2",
                "P1 holds R2"
                "P1 requests R1",
                "P1 holds R1", 
                "P0 requests R0",
                "P0 holds R0", 
                "P1 requests R1", 
                "P1 holds R1"
            ]
        
        # reset matrices and edges
        self.matrixAlloc = np.zeros((self.numberProcesses, self.numberResources), dtype=int)
        self.requestMatrix = np.zeros((self.numberProcesses, self.numberResources), dtype=int)
        self.availableResources = np.array(self.resourceInstance.copy(), dtype=int)
        self.requestEdge = []
        self.claimEdge = []
        self.deadlockedProcesses = []

    def simulate(self):
        # run the simulation based on the statement list
        plt.rcParams['toolbar'] = 'None'  
        plt.axis('off')  
        plt.ion()  
        plt.figure(figsize=(10, 6))  
        plt.show()  
        
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
            if self.detectDeadlock():
                print("Deadlock detected!")
                print(f"Deadlocked processes: {', '.join(f'p{i}' for i in self.deadlockedProcesses)}")
                if len(self.deadlockedProcesses) == self.numberProcesses:
                    self.system_deadlocked = True
                    print("System completely deadlocked, halting program")
            
            # print matrix states
            self.printMatrixState()
            
            # show graph for each statement
            self.drawGraph()

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
            
            # for graph visualization, add request edge
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
                    
                    print(f"p{processNum} now holds {self.matrixAlloc[processNum][resourceNum]} instances of R{resourceNum}")
                else:
                    print(f"Error: No available instances of R{resourceNum}")
            else:
                print(f"Error: p{processNum} didn't request R{resourceNum}")
        
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
                
                print(f"P{processNum} released R{resourceNum}, now holds {self.matrixAlloc[processNum][resourceNum]} instances")
                
                # check if any process is waiting for this resource
                self.checkPendingRequests(resourceNum)
            else:
                print(f"Error: p{processNum} doesn't hold any instances of R{resourceNum}")
        
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
                
                print(f"P{i} was granted R{resourceNum} from waiting queue, now holds {self.matrixAlloc[i][resourceNum]} instances")  # Fixed: "holds" to "granted"
                
                # only grant to one process (this can be modified if needed)
                return

    def detectDeadlock(self):
        # create a directed graph
        graph = nx.DiGraph()
        
        # Add nodes for processes and resources
        processes = list(range(self.numberProcesses))
        resources = list(range(self.numberProcesses, self.numberProcesses + self.numberResources))
        graph.add_nodes_from(processes + resources)
        
        # Add all edges (request edges and claim edges)
        graph.add_edges_from(self.requestEdge + self.claimEdge)
        
        # check for cycles in the graph
        # a cycle in a resource allocation graph indicates a deadlock
        self.deadlockedProcesses = []
        for cycle in nx.simple_cycles(graph):
            for node in cycle:
                # Only add process nodes to deadlocked processes list
                if node < self.numberProcesses and node not in self.deadlockedProcesses:
                    self.deadlockedProcesses.append(node)
                    
        # Return True if there are deadlocked processes
        return len(self.deadlockedProcesses) > 0

    def printMatrixState(self):
        # print the current state of matrices
        print("\nCurrent State:")
        print("Allocation Matrix:")
        printLabels(self.matrixAlloc)
        print("\nRequest Matrix:")
        printLabels(self.requestMatrix)
        print("\nAvailable Resources:")
        print(self.availableResources)

    def drawGraph(self):
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
            labels[j] = f'R{resource_id}\n({allocated}/{total})'
        
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
            if len(self.deadlockedProcesses) == self.numberProcesses:
                plt.title(f"Multiple Instance Resource Allocation Graph\nDeadlock Detected!", 
                     color='red', fontsize=12)
        else:
            plt.title("Multiple Instance Resource Allocation Graph (No Deadlock Detected)", fontsize=12)

        
        # make sure axis stays gone
        plt.axis("off")
        plt.tight_layout()
        plt.pause(1)
        plt.savefig(f"{self.step + 1}.png")
        
        # if this is the final step, prompt for shutdown
        if self.step == len(self.statementsList):
            self.shutdownPrompt()

    def shutdownPrompt(self):
        # prompt for program shutdown
        input("Press enter to exit...")

# example usage
if __name__ == '__main__':
    # create a manager
    rm = MultipleInstanceResourceManager(3, 2, [2, 2, 2])
    
    # choose a scenario
    scenario = input("Enter scenario (1 = deadlock, 2 = noDeadlock): ") 
    if scenario == '1':
        rm.scenario("deadlock")
    elif scenario == '2':
        rm.scenario("noDeadlock")
    else:
        print("Invalid scenario. Defaulting to deadlock scenario.")
        rm.scenario("deadlock")
    
    # run the simulation
    rm.simulate()