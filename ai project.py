import tkinter as tk
from tkinter import ttk, scrolledtext
import matplotlib.pyplot as plt
import heapq
import time
import networkx as nx  # For graph visualization

# Expanded health problem graph for risk assessment
# Nodes: Symptoms and diagnoses with risk levels (e.g., Low, Medium, High risk of complications)
# Edges: Costs (e.g., time to assess) and heuristics (estimated risk reduction)
graph = {
    'Start': {'Fever': {'cost': 1, 'heuristic': 4}, 'Cough': {'cost': 1, 'heuristic': 3}, 'Sneezing': {'cost': 1, 'heuristic': 2}, 'Headache': {'cost': 1, 'heuristic': 3}},
    'Fever': {'Flu': {'cost': 1, 'heuristic': 2}, 'Cold': {'cost': 1, 'heuristic': 1}},
    'Cough': {'Flu': {'cost': 1, 'heuristic': 2}, 'Cold': {'cost': 1, 'heuristic': 1}},
    'Sneezing': {'Allergies': {'cost': 1, 'heuristic': 1}},
    'Headache': {'Migraine': {'cost': 1, 'heuristic': 1}, 'Diabetic_Retinopathy': {'cost': 2, 'heuristic': 3}},  # Added for risk example
    'Flu': {'High_Risk': {'cost': 1, 'heuristic': 0}},  # Risk of complications
    'Cold': {'Low_Risk': {'cost': 1, 'heuristic': 0}},
    'Allergies': {'Medium_Risk': {'cost': 1, 'heuristic': 0}},
    'Migraine': {'Medium_Risk': {'cost': 1, 'heuristic': 0}},
    'Diabetic_Retinopathy': {'High_Risk': {'cost': 1, 'heuristic': 0}},  
    'Low_Risk': {},
    'Medium_Risk': {},
    'High_Risk': {}
}

# Heuristics for Best First and A*
heuristics = {
    'Start': 5,
    'Fever': 3,
    'Cough': 3,
    'Sneezing': 2,
    'Headache': 3,
    'Flu': 2,
    'Cold': 1,
    'Allergies': 1,
    'Migraine': 1,
    'Diabetic_Retinopathy': 3,
    'Low_Risk': 0,
    'Medium_Risk': 0,
    'High_Risk': 0
}

# Best First Search
def best_first_search(start, goal):
    frontier = [(heuristics[start], start)]
    came_from = {start: None}
    cost_so_far = {start: 0}
    nodes_expanded = 0
    
    while frontier:
        _, current = heapq.heappop(frontier)
        nodes_expanded += 1
        if current == goal:
            break
        for neighbor, data in graph[current].items():
            new_cost = cost_so_far[current] + data['cost']
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = heuristics[neighbor]
                heapq.heappush(frontier, (priority, neighbor))
                came_from[neighbor] = current
    return reconstruct_path(came_from, start, goal), nodes_expanded

# Beam Search (beam width = 2)
def beam_search(start, goal, beam_width=2):
    frontier = [(heuristics[start], start)]
    came_from = {start: None}
    cost_so_far = {start: 0}
    nodes_expanded = 0
    
    while frontier:
        current_frontier = []
        for _ in range(min(beam_width, len(frontier))):
            _, current = heapq.heappop(frontier)
            nodes_expanded += 1
            if current == goal:
                return reconstruct_path(came_from, start, goal), nodes_expanded
            for neighbor, data in graph[current].items():
                new_cost = cost_so_far[current] + data['cost']
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = heuristics[neighbor]
                    current_frontier.append((priority, neighbor))
                    came_from[neighbor] = current
        frontier = sorted(current_frontier)[:beam_width]
    return None, nodes_expanded

# A* Search
def a_star_search(start, goal):
    frontier = [(0 + heuristics[start], start)]
    came_from = {start: None}
    cost_so_far = {start: 0}
    nodes_expanded = 0
    
    while frontier:
        _, current = heapq.heappop(frontier)
        nodes_expanded += 1
        if current == goal:
            break
        for neighbor, data in graph[current].items():
            new_cost = cost_so_far[current] + data['cost']
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + heuristics[neighbor]
                heapq.heappush(frontier, (priority, neighbor))
                came_from[neighbor] = current
    return reconstruct_path(came_from, start, goal), nodes_expanded

# AO* Search (simplified for OR graph; full AND-OR is complex)
def ao_star_search(start, goal):
    # Treating as OR graph, similar to A*
    return a_star_search(start, goal)

def reconstruct_path(came_from, start, goal):
    current = goal
    path = []
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start)
    path.reverse()
    return path

# Tkinter GUI with Beautiful and Professional Blue-Themed Styling 
class HealthAISearchApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Health AI Search for Risk Assessment")
        self.root.geometry("900x700")
        self.root.configure(bg='#E3F2FD')  # Light blue background for a calm, professional feel
        
        # Apply ttk style for professional look
        style = ttk.Style()
        style.theme_use('clam')  # Modern theme
        
        # Configure colors for a professional blue-themed interface (no green)
        style.configure('TLabel', background='#E3F2FD', foreground='#0D47A1', font=('Arial', 12, 'bold'))  # Dark blue text
        style.configure('TButton', background='#1565C0', foreground='white', font=('Arial', 10, 'bold'), relief='raised')  # Navy blue buttons
        style.map('TButton', background=[('active', '#42A5F5')])  # Light blue on hover
        style.configure('TCombobox', fieldbackground='white', background='#BBDEFB', font=('Arial', 10))  # Light blue combo
        style.configure('TFrame', background='#E3F2FD')
        
        # Main frame for layout
        main_frame = ttk.Frame(root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title and description
        title_label = ttk.Label(main_frame, text="AI Search for Health Risk Assessment", font=('Arial', 16, 'bold'), foreground='#0D47A1')
        title_label.pack(pady=10)
        desc_label = ttk.Label(main_frame, text="Inspired by DeepMind Health: Assess risk from symptoms to diagnoses like Diabetic Retinopathy.", font=('Arial', 10))
        desc_label.pack(pady=5)
        
        # Algorithm selection
        algo_frame = ttk.Frame(main_frame, style='TFrame')
        algo_frame.pack(pady=10)
        ttk.Label(algo_frame, text="Select Algorithm:").pack(side=tk.LEFT, padx=5)
        self.algorithm = tk.StringVar()
        algorithms = ["Best First Search", "Beam Search", "A* Search", "AO* Search"]
        self.algo_combo = ttk.Combobox(algo_frame, textvariable=self.algorithm, values=algorithms, state='readonly')
        self.algo_combo.pack(side=tk.LEFT)
        
        # Goal selection
        goal_frame = ttk.Frame(main_frame, style='TFrame')
        goal_frame.pack(pady=10)
        ttk.Label(goal_frame, text="Select Risk Goal:").pack(side=tk.LEFT, padx=5)
        self.goal = tk.StringVar()
        goals = ["Low_Risk", "Medium_Risk", "High_Risk"]
        self.goal_combo = ttk.Combobox(goal_frame, textvariable=self.goal, values=goals, state='readonly')
        self.goal_combo.pack(side=tk.LEFT)
        
        # Run button
        self.run_btn = ttk.Button(main_frame, text="Run Search", command=self.run_search)
        self.run_btn.pack(pady=15)
        
        # Results text area with styled background
        self.results_text = scrolledtext.ScrolledText(main_frame, width=90, height=12, bg='#F5F5F5', fg='#0D47A1', font=('Courier', 10), relief='sunken', bd=2)
        self.results_text.pack(pady=10)
        
        # Graph button
        self.graph_btn = ttk.Button(main_frame, text="Show Analysis Graphs", command=self.show_graphs)
        self.graph_btn.pack(pady=10)
        
        self.path = None
        self.nodes_expanded = 0
        self.times = {}
    
    def run_search(self):
        start = 'Start'
        goal = self.goal.get()
        algo = self.algorithm.get()
        
        if not goal or not algo:
            self.results_text.insert(tk.END, "Please select algorithm and goal.\n")
            return
        
        start_time = time.time()
        if algo == "Best First Search":
            self.path, self.nodes_expanded = best_first_search(start, goal)
        elif algo == "Beam Search":
            self.path, self.nodes_expanded = beam_search(start, goal)
        elif algo == "A* Search":
            self.path, self.nodes_expanded = a_star_search(start, goal)
        elif algo == "AO* Search":
            self.path, self.nodes_expanded = ao_star_search(start, goal)
        end_time = time.time()
        
        self.times[algo] = end_time - start_time
        
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"Algorithm: {algo}\n")
        self.results_text.insert(tk.END, f"Path to Risk Assessment: {' -> '.join(self.path)}\n")
        self.results_text.insert(tk.END, f"Nodes Expanded: {self.nodes_expanded}\n")
        self.results_text.insert(tk.END, f"Time Taken: {self.times[algo]:.4f} seconds\n")
        self.results_text.insert(tk.END, "This path shows symptom assessment leading to risk level, aiding timely intervention.\n")
    
    def show_graphs(self):
        if not self.times:
            return
        
        # Performance comparison bar chart with blue colors
        algos = list(self.times.keys())
        times = list(self.times.values())
        
        plt.figure(figsize=(8, 5), facecolor='#E3F2FD')
        bars = plt.bar(algos, times, color=['#1565C0', '#42A5F5', '#BBDEFB', '#90CAF9'])  # Blue shades
        plt.xlabel('Algorithm', color='#0D47A1')
        plt.ylabel('Time (seconds)', color='#0D47A1')
        plt.title('Algorithm Performance Comparison', color='#0D47A1')
        plt.xticks(color='#0D47A1')
        plt.yticks(color='#0D47A1')
        plt.gca().set_facecolor('#F5F5F5')
        plt.show()
        
        # Graph structure visualization with blue colors
        G = nx.DiGraph()
        for node, neighbors in graph.items():
            for neighbor in neighbors:
                G.add_edge(node, neighbor)
        
        plt.figure(figsize=(10, 8), facecolor='#E3F2FD')
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='#BBDEFB', node_size=1500, font_size=8, font_weight='bold', font_color='#0D47A1', edge_color='#42A5F5')
        plt.title('Health Risk Assessment Graph', color='#0D47A1')
        plt.gca().set_facecolor('#F5F5F5')
        plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    app = HealthAISearchApp(root)
    root.mainloop() 
