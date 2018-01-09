from components import *
from graphviz import Digraph
import networkx as nx
from terminaltables import SingleTable
from pint import UnitRegistry
import yaml
import json
from warnings import warn
from datetime import datetime, timedelta
from copy import deepcopy
import plotly as py
import plotly.figure_factory as ff
from plotly.colors import DEFAULT_PLOTLY_COLORS as colors
from datetime import datetime, timedelta

class Apparatus(object):
	id_counter = 0
	def __init__(self, name=None):
		self.network = []
		self.components = set()
		# if given a name, then name the appartus, else default to a sequential name
		if name is not None:
			self.name = name
		else:
			self.name = "Apparatus_" + str(Apparatus.id_counter)
			Apparatus.id_counter += 1
	
	def add(self, from_component, to_component, tube):
		'''add a connection in the apparatus'''
		assert issubclass(from_component.__class__, Component)
		assert issubclass(to_component.__class__, Component)
		assert issubclass(tube.__class__, Tube)
		
		self.network.append((from_component, to_component, tube))
		self.components.update([from_component, to_component])

	def visualize(self, title=True, label_tubes=False, node_attr={}, edge_attr={}, graph_attr=dict(splines="ortho",  nodesep="1"), format="pdf", filename=None):
		'''generate a visualization of the graph of an apparatus'''
		self.compile() # ensure apparatus is valid
		f = Digraph(name=self.name, 
					node_attr=node_attr, 
					edge_attr=edge_attr, 
					graph_attr=graph_attr, 
					format=format, 
					filename=filename)

		# go from left to right adding components and their tubing connections
		f.attr(rankdir='LR')
		f.attr('node', shape='circle')
		for x in self.network:
			tube_label = f"Length {x[2].length}\nID {x[2].inner_diameter}\nOD {x[2].outer_diameter}" if label_tubes else ""
			f.edge(x[0].name, x[1].name, label=tube_label)

		# show the title of the graph
		if title:
			title = title if title != True else self.name
			f.attr(label=title)

		f.view(cleanup=True)

	def summarize(self):
		'''print a summary table of the apppartus'''
		self.compile() # ensure apparatus is valid
		summary = [["Name", "Type", "Address"]] # header rows of components table
		for component in list(self.components):
			component_summary = [component.name, component.__class__.__name__]
			if issubclass(component.__class__, ActiveComponent): # add the address if it has one
				component_summary.append(component.address)
			else:
				component_summary.append("")
			summary.append(component_summary)

		# generate the components table
		table = SingleTable(summary)
		table.title = "Components"
		print(table.table)

		# store and calculate the computed totals for tubing
		total_length = 0 * ureg.mm
		total_volume = 0 * ureg.ml
		for tube in [x[2] for x in self.network]:
			total_length += tube.length
			total_volume += tube.volume

		# summarize the tubing
		summary = [["From", "To", "Length", "Inner Diameter", "Outer Diameter", "Volume", "Temp"]] # header row
		for edge in self.network:
			summary.append([edge[0].name, 
							edge[1].name, 
							round(edge[2].length, 4), 
							round(edge[2].inner_diameter, 4), 
							round(edge[2].outer_diameter, 4), 
							round(edge[2].volume.to("ml"), 4)])
			if edge[2].temp is not None:
				summary[-1].append(round(edge[2].temp, 4))
			else:
				summary[-1].append(None)
		summary.append(["", "Total", round(total_length, 4), "n/a", "n/a", round(total_volume.to("ml"), 4), "n/a"]) # footer row

		# generate the tubing table
		table = SingleTable(summary)
		table.title = "Tubing"
		table.inner_footing_row_border = "True"
		print(table.table)	

	def __repr__(self):
		return self.name	

	def compile(self):
		'''make sure that the apparatus is valid'''
		G = nx.Graph() # convert the network to an undirected NetworkX graph
		G.add_edges_from([(x[0], x[1]) for x in self.network])
		if not nx.is_connected(G): # make sure that all of the components are connected
			raise ValueError("Unable to compile: not all components connected")

		# valve checking
		for valve in list(set([x[0] for x in self.network if issubclass(x[0].__class__, Valve)])):
			for name in valve.mapping.keys():
				# ensure that valve's mapping components are part of apparatus
				if name not in valve.used_names:
					raise ValueError(f"Invalid mapping for Valve {valve}. No component named {name} exists.")
			# no more than one output from a valve (might have to change this)
			if len([x for x in self.network if x[0] == valve]) != 1:
				raise ValueError(f"Valve {valve} has multiple outputs.")

			# make sure valve's mapping is complete
			non_mapped_components = [x[0] for x in self.network if x[1] == valve and valve.mapping.get(x[0].name) is None]
			if non_mapped_components:
				raise ValueError(f"Valve {valve} has incomplete mapping. No mapping for {non_mapped_components}")

		return True

class Protocol(object):
	id_counter = 0
	def __init__(self, apparatus, duration=None, name=None):
		assert type(apparatus) == Apparatus
		if apparatus.compile(): # ensure apparatus is valid
			self.apparatus = apparatus
		self.procedures = []
		if name is not None:
			self.name = name
		else:
			self.name = "Protocol_" + str(Protocol.id_counter)
			Protocol.id_counter += 1

		# check duration, if given
		if duration not in [None, "auto"]:
			duration = ureg.parse_expression(duration)
			if duration.dimensionality != ureg.hours.dimensionality:
				raise ValueError("Incorrect dimensionality for duration. Must be a unit of time such as \"seconds\" or \"hours\".")
		self.duration = duration

	def _is_valid_to_add(self, component, **kwargs):
		# make sure that the component being added to the protocol is part of the apparatus
		if component not in self.apparatus.components:
			raise ValueError(f"{component} is not a component of {self.apparatus.name}.")

		# check that the keyword is a valid attribute of the component
		if not component.is_valid_attribute(**kwargs):
			raise ValueError(f"Invalid attributes present for {component.name}.")
		
	def add(self, component, start_time="0 seconds", stop_time=None, **kwargs):
		'''add a procedure to the protocol for an apparatus'''

		# make sure the component is valid to add
		self._is_valid_to_add(component, **kwargs)

		# parse the start and stop times if given
		start_time = ureg.parse_expression(start_time)
		if stop_time is None and self.duration is None:
			raise ValueError("Must specify protocol duration during instantiation in order to omit stop_time. " \
				"To automatically set duration as end of last procedure in protocol, use duration=\"auto\".")
		elif stop_time is not None:
			stop_time = ureg.parse_expression(stop_time)

		# perform the mapping for valves
		if issubclass(component.__class__, Valve) and kwargs.get("setting") is not None:
			kwargs["setting"] = component.mapping[kwargs["setting"]]

		# add the procedure to the procedure list
		self.procedures.append(dict(start_time=start_time, stop_time=stop_time, component=component, params=kwargs))

	def compile(self):

		output = {}

		# infer the duration of the protocol
		if self.duration == "auto":
			self.duration = sorted([x["stop_time"] for x in self.procedures], key=lambda z: z.to_base_units().magnitude if type(z) == ureg.Quantity else 0)
			if all([x == None for x in self.duration]):
				raise ValueError("Unable to automatically infer duration of protocol. Must define stop_time for at least one procedure to use duration=\"auto\".")
			self.duration = self.duration[-1]

		
		for component in [x for x in self.apparatus.components if issubclass(x.__class__, ActiveComponent)]:
			# make sure all active components are activated, raising warning if not
			if component not in [x["component"] for x in self.procedures]:
				warn(f"{component} is an active component but was not used in this procedure. If this is intentional, ignore this warning.")

			# determine the procedures for each component
			component_procedures = sorted([x for x in self.procedures if x["component"] == component], key=lambda x: x["start_time"])

			# skip compilation of components with no procedures added
			if not len(component_procedures):
				continue

			# check for conflicting continuous procedures
			if len([x for x in component_procedures if x["start_time"] is None and x["stop_time"] is None]) > 1:
				raise ValueError((f"{component} cannot have two procedures for the entire duration of the protocol. " 
					"If each procedure defines a different attribute to be set for the entire duration, combine them into one call to add(). "  
					"Otherwise, reduce ambiguity by defining start and stop times for each procedure."))

			for i, procedure in enumerate(component_procedures):
				# ensure that the start time is before the stop time if given
				if procedure["stop_time"] is not None and procedure["start_time"] > procedure["stop_time"]:
					raise ValueError("Start time must be less than or equal to stop time.")

				# make sure that the start time isn't outside the duration
				if self.duration is not None and procedure["start_time"] is not None and procedure["start_time"] > self.duration:
					raise ValueError(f"Procedure cannot start at {procedure['start_time']}, which is outside the duration of the experiment ({self.duration}).")

				# make sure that the end time isn't outside the duration
				if self.duration is not None and procedure["stop_time"] is not None and procedure["stop_time"] > self.duration:
					raise ValueError(f"Procedure cannot end at {procedure['stop_time']}, which is outside the duration of the experiment ({self.duration}).")
				
				# automatically infer start and stop times
				try:
					if component_procedures[i+1]["start_time"] == ureg.parse_expression("0 seconds"):
						raise ValueError(f"Ambiguous start time for {procedure['component']}.")
					elif component_procedures[i+1]["start_time"] is not None:
						procedure["stop_time"] = component_procedures[i+1]["start_time"]
				except IndexError:
					if procedure["stop_time"] is None:
						procedure["stop_time"] = self.duration 

			output[component] = component_procedures

		return output

	def yaml(self):
		compiled = deepcopy(self.compile())
		for item in compiled.items():
			for procedure in item[1]:
				procedure["start_time"] = procedure["start_time"].to_timedelta()
				procedure["stop_time"] = procedure["stop_time"].to_timedelta()
				del procedure["component"]
		compiled = {str(k): v for (k, v) in compiled.items()}
		return yaml.dump(compiled)

	def json(self):
		compiled = deepcopy(self.compile())
		for item in compiled.items():
			for procedure in item[1]:
				procedure["start_time"] = str(procedure["start_time"].to_timedelta())
				procedure["stop_time"] = str(procedure["stop_time"].to_timedelta())
				del procedure["component"]
		compiled = {str(k): v for (k, v) in compiled.items()}
		return json.dumps(compiled, indent=4, sort_keys=True)

	def visualize(self):
		df = []
		for component, procedures in self.compile().items():
			for procedure in procedures:
				df.append(dict(
					Task=str(component),
					Start=str(datetime(2000, 1, 1) + procedure["start_time"].to_timedelta()),
					Finish=str(datetime(2000, 1, 1) + procedure["stop_time"].to_timedelta()),
					Resource=str(procedure["params"])))
		df.sort(key=lambda x: x["Task"])

		# ensure that color coding keeps color consistent for params
		colors_dict = {}
		color_idx = 0
		for params in list(set([str(x["params"]) for x in self.procedures])):
			colors_dict[params] = colors[color_idx % len(colors)]
			color_idx += 1

		# create the graph
		fig = ff.create_gantt(df, group_tasks=True, colors=colors_dict, index_col='Resource', showgrid_x=True, title=self.name)

		# add the hovertext
		for i in range(len(fig["data"])):
			fig["data"][i].update(text=df[i]["Resource"], hoverinfo="text")

		# plot it
		py.offline.plot(fig, filename=f'{self.name}.html')