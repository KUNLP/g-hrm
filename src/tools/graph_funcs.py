import networkx as nx
import numpy as np
import pandas as pd
import pickle
import os

from IPython import embed
import json
import re
from typing import List, Dict, Any, Optional

class graph_funcs():
    # init
    def __init__(self, graph):
        self._reset(graph)
        self.graph = graph
        self.integrated_solver = None  # Will be set by GraphAgent

    def _reset(self, graph):
        graph_index = {}
        nid_set = set()
        for node_type in graph:
            for nid in graph[node_type]:
                assert nid not in nid_set
                nid_set.add(nid)
                graph_index[nid] = graph[node_type][nid]
        self.graph_index = graph_index

    def check_neighbours(self, node, neighbor_type=None):
        if neighbor_type:
            # Edge type 표준화
            standardized_relation = self._standardize_relation(neighbor_type)
            try:
                neighbors = self.graph_index[node]['neighbors'][standardized_relation]
                # 리스트를 그대로 반환 (문자열로 변환하지 않음)
                return neighbors if isinstance(neighbors, list) else [neighbors]
            except KeyError:
                # 원본 relation도 시도
                try:
                    neighbors = self.graph_index[node]['neighbors'][neighbor_type]
                    return neighbors if isinstance(neighbors, list) else [neighbors]
                except KeyError:
                    return f"Edge type '{neighbor_type}' not found for node {node}"
        else:
            neighbors = self.graph_index[node]['neighbors']
            return neighbors if isinstance(neighbors, list) else [neighbors]
    
    def _standardize_relation(self, relation):
        """실제 그래프의 edge type 패턴에 맞게 표준화"""
        # 실제 그래프는 "Entity1-relation-Entity2" 형태
        relation_mapping = {
            'treats': 'Compound-treats-Disease',
            'includes': 'Pharmacologic_Class-includes-Compound', 
            'causes': 'Disease-causes-Symptom',
            'interacts': 'Compound-interacts-Compound',
            'localizes': 'Disease-localizes-Anatomy',
            'presents': 'Disease-presents-Symptom',
            'associates': 'Disease-associates-Gene',
            'binds': 'Compound-binds-Gene',
            'participates': 'Gene-participates-Biological Process',
            'downregulates': 'Disease-downregulates-Gene',
            'upregulates': 'Disease-upregulates-Gene'
        }
        return relation_mapping.get(relation.lower(), relation)

    # check the attributes of the nodes
    def check_nodes(self, node, feature=None):
        if feature:
            try:
                # Check if node exists first
                if node not in self.graph_index:
                    return f"Node {node} not found in graph"
                
                # Try to get the specific feature
                if feature in self.graph_index[node]['features']:
                    feature_value = self.graph_index[node]['features'][feature]
                    # Feature 함수 강화: 항상 실제 이름 반환
                    if feature == 'name' and feature_value:
                        return str(feature_value)
                    else:
                        return str(feature_value)
                else:
                    # Feature not found - return clear error message
                    return f"Feature '{feature}' not found for node {node}"
                    
            except Exception as e:
                return f"Error accessing feature '{feature}' for node {node}: {str(e)}"
        else:
            try:
                if node not in self.graph_index:
                    return f"Node {node} not found in graph"
                return str(self.graph_index[node]['features'])
            except Exception as e:
                return f"Error accessing features for node {node}: {str(e)}"
    
    def get_node_name(self, node):
        """강화된 Feature 함수: 항상 node의 name을 반환"""
        try:
            if node not in self.graph_index:
                return f"Node {node} not found in graph"
            
            features = self.graph_index[node]['features']
            if 'name' in features:
                return str(features['name'])
            else:
                return f"No name feature found for node {node}"
        except Exception as e:
            return f"Error getting name for node {node}: {str(e)}"
    
    # IntegratedSolver tool methods
    def solve_question_directly(self, question: str) -> str:
        """Solve question directly using IntegratedSolver"""
        if hasattr(self, 'integrated_solver') and self.integrated_solver:
            try:
                return self.integrated_solver.solve_question(question)
            except Exception as e:
                return f"Error solving question: {str(e)}"
        else:
            return "IntegratedSolver not available"

    def solve_pharmacologic_class(self, condition: str) -> str:
        """Solve pharmacologic class question"""
        if hasattr(self, 'integrated_solver') and self.integrated_solver:
            try:
                return self.integrated_solver.solve_pharmacologic_class(condition)
            except Exception as e:
                return f"Error solving pharmacologic class: {str(e)}"
        else:
            return "IntegratedSolver not available"

    def solve_cellular_component(self, condition: str, regulation: str = "upregulated") -> str:
        """Solve cellular component question"""
        if hasattr(self, 'integrated_solver') and self.integrated_solver:
            try:
                return self.integrated_solver.solve_cellular_component(condition, regulation)
            except Exception as e:
                return f"Error solving cellular component: {str(e)}"
        else:
            return "IntegratedSolver not available"

    def solve_pathway(self, condition: str, regulation: str = "upregulated") -> str:
        """Solve pathway question"""
        if hasattr(self, 'integrated_solver') and self.integrated_solver:
            try:
                return self.integrated_solver.solve_pathway(condition, regulation)
            except Exception as e:
                return f"Error solving pathway: {str(e)}"
        else:
            return "IntegratedSolver not available"

    def solve_gene_count(self, gene_name: str) -> str:
        """Solve gene count question"""
        if hasattr(self, 'integrated_solver') and self.integrated_solver:
            try:
                return self.integrated_solver.solve_gene_count(gene_name)
            except Exception as e:
                return f"Error solving gene count: {str(e)}"
        else:
            return "IntegratedSolver not available"

    def solve_disease_count(self, disease_name: str) -> str:
        """Solve disease count question"""
        if hasattr(self, 'integrated_solver') and self.integrated_solver:
            try:
                return self.integrated_solver.solve_disease_count(disease_name)
            except Exception as e:
                return f"Error solving disease count: {str(e)}"
        else:
            return "IntegratedSolver not available"
    
    def check_degree(self, node, neighbor_type):
        return str(len(self.graph_index[node]['neighbors'][neighbor_type]))
    
    def check_all_neighbour(self, node_q):
        nodes = {}
        for node, node_info in self.graph_index.items():
            for neighbor_type, neighbours in node_info['neighbors'].items():
                if node_q in neighbours:
                    nodes[node] = neighbor_type 
        return nodes
    
    def reverse_neighbor(self, node_id, edge_type):
        """Get nodes that have this node as a neighbor with the specified edge type (reverse lookup)"""
        if node_id not in self.graph_index:
            return f"Node {node_id} not found in graph"
        
        reverse_neighbors = []
        for potential_neighbor_id, neighbor_data in self.graph_index.items():
            if potential_neighbor_id == node_id:
                continue
            for target_id, edges in neighbor_data.get('neighbors', {}).items():
                if target_id == node_id:
                    for edge in edges:
                        if edge['edge_type'] == edge_type:
                            reverse_neighbors.append(potential_neighbor_id)
        
        if not reverse_neighbors:
            return f"No nodes found with '{edge_type}' relationship pointing to {node_id}"
        
        return reverse_neighbors

