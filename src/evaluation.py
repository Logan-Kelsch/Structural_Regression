'''
	Logan Kelsch - 2/26/26

	Holy moly took long enough to get started on this file.
	This file will contain all overarching and helper functionality in the evaluation of genes and USAGE of gene instantiation in our framework.
	this file will also contain the evaluation of grammar
	this file will also contain the evaluation of a population in terms of itself

	In this text block here I will develop the instructure.

BASIC:

	We ultimately will have some 1d array of length N (or 2d array if we split along days, I'm sure we will)
	This vector will be our solutions that will be interpreted somehow
			expanding on "solutions" and "interpreted somehow"

	gene evaluation will provide information for selecting parent / parent quality as well as iterating grammar
	grammar evaluation will provide information that goes hand in hand with gene evaluation for iterating grammar
	population evaluation will provide information that removes redundancy or excessive inbreeding in a population


solutions:

	We will have several different angles that we could try to solve for, which I will write in likely order of development.
	A main objective of developing this consists of ensuring a modular approach for indiana jones style swapping of methods.

	- Boolean anomoly detection
			this allows us to interpret positive values as true prediction (ex: price moves 3st up from 20 candle EMA)
	- Moving average SD
			this allows for predictions to be made regarding directionality, while holding a normalized output
	- Volume moving average SD
			This allows for more generalized predictions to be made regarding behavior of volume (directionality removed)
	- certainly some more approaches 

Mod dev step:
	
	The first steps will be:
	- creating a solution array for boolean anomoly detection
	- creating a gene matrix evaluation functionality for 2d instantiated gene matrices
			these can be made modular by having a class or object mechanism that ties evaluation style mechanism to solution instantiation mechanism
			for example, for the boolean anomoly, classification approaches will not be as effective, so we will have a cost score
				in which considers the difference of that and the expected value of this solution space.
				this will allow for a negative score to be undesirable, zero to be equal to market performance, and >>0 to be ideal.
				Expected cost = c_FP x FP + c_FN x FN


'''

import initialization as _I
import numpy as np

class Solver:
	'''
	This class object will simply contain data that is wanted to resolve the solution vector from given data
	'''
	def __init__(
		self,
		population	:	_I.Population,
		t_vec		:	str	=	'Close', #could be volume
		t_mode		:	str	=	'AD', #AD for anomoly detection, or RE for raw emission
		emission	:	list=	['some operations here that can be turned into op list or instantiated easily'],
		AD_cond		:	tuple=	('lt', -2)
	):
		'''emission is interpreted as functions applied to t_vec from left to right.'''

		self._tmode= t_mode
		self._emission= emission
		self._AD_cond = AD_cond

		#match case for transforming t_vec string into a column index in population variable
		target_idx = -1
		match(t_vec):
			#for logical cases we need to see if the time variables are 
			#added as terminal states to interpret what locations they are at.
			case 'Volume':
				#NOTE this is assuming that THLCV or THLOCV is the standard shape of data coming
				if(population._time_terminals):
					#walking backwards is dow(-1), tod(-2), vol(-3), close(-4)
					target_idx = population._T_idx[-3]
				else:
					#walking backwards is vol(-1), close(-2)
					target_idx = population._T_idx[-1]
			case 'Close':
				#NOTE this is assuming that THLCV or THLOCV is the standard shape of data coming
				if(population._time_terminals):
					#walking backwards is dow(-1), tod(-2), vol(-3), close(-4)
					target_idx = population._T_idx[-4]
				else:
					#walking backwards is vol(-1), close(-2)
					target_idx = population._T_idx[-2]
			case _:
				raise ValueError(f'Target vector (t_vec parameter in Solver Initialization) of "{t_vec}" dont make noooo sense. Try "Close".')
			
		if(target_idx<0):
			raise ValueError('Did not form a good target vector index in solver initialization. Got idx (-1)')
		
		self._tidx = target_idx
		
	def solve(
		self,
		Population :	_I.Population
	):
		'''Solves target vector and mask. solution mask is where a conditional statement is true.
		raw emission is used directly in computation for some form of emission prediction,
		where raw emission[solution mask] is used in cost based evaluation function'''

		#now we have the tvec index and can select it for computation with Population._X_inst[:, target_idx]
		if(self._tmode=='AD'|self._tmode=='RE'):
			
			#Here we need to generate the raw emission
			raw_emission = generate_raw_emission(Population, self._tidx, self._emission)

			#generate a mask for where we want to allow comparison to be done
			#the default will be at all locations
			solution_mask = np.full(raw_emission.shape, True, dtype=bool)


			if(self._tmode=='AD'):
				
				#then we need to make a boolean masking variable where raw emission is true under AD_cond parameter interpretation
				solution_mask = generate_solution_mask(raw_emission, self._AD_cond)	

		else:
			raise NotImplementedError(f'Target Mode of "{self._tmode}" is not supported at this moment.')
		
		#now we have successfully solved the target for the population
		#returns returns read all about it
		return raw_emission, solution_mask

		

#first candidate evaluation function should be a light modular somewhat function that containst the first solution type selection

def evaluate(
	population	:	_I.Population,
	raw_emissions:	any,
	solution_mask:	any	
):
	#bring in population to be able to see the instantiated genes
	#identify the column that has real data that the solution will be derived from
	#call generate_solution() to get the desired vector for comparison with the selected function

	#NOTE SOMETHING LIKE
	#
	#	x[solution_mask].sum() for cost based evaluation, as initial example?? not really just scrap comments at the moment.
	#
	#

	#NOTE IF SOLVER._T_MODE IS 'AD' THEN WE MUST CONVERT POPULATION INSTANTIATION INTO BOOLEAN COLUMNS WHERE G_mn = G_mn > 0
	return


def generate_raw_emission(
	Population	:	_I.Population,
	target_idx	:	int,
	emission	:	list,
):
	return #something

def generate_solution_mask(
	raw_emission:	any,
	AD_cond		:	tuple
):
	return #something