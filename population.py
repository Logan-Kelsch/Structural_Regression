import transform as t



class Chromosome:
	'''
	As far as I am now, this class should contain nothing but 
	instructions (gene sequence)
	'''

	def __init__(
		self,
		genes	:	list	=	[],
		length	:	int		=	0
	)	->	None:
		self.genes  = genes
		self.length = length


class gene:
	'''
	As far as I am now, this class shoudl contain nothing but
	instruction encrypting

	APPLICATION SPECIFIC NOTATIONS (IN ORDER)
	
	1.
		F - function, followed by ID

		
	3.
		x,a,k,d,d_ - parameters:
			x/a  - Can be:
				T - terminal (sensor), followed by raw column index
				G - gene     (sensor), followed by g index
			k    - Can be: float
			d/dd - Can be: float


	example sequence of genes that consist of 
	g0: 1st raw feature
	g1: 3rd raw feature
	g2: 2nd raw feature
	g3: average of g2 over window 54
	g4: sum of g1 and g3
	g5: hawkes process function on g4 with kappa of 0.232
	g6: min-max-norm range of g5 over max23 and min7

	g1: 'F0T2'
	g2: 'F0T1'
	g3: 'F6G2d54'
	g4: 'F7G1G3'
	g5: 'F9G4k0.232'
	g6: 'F8G5d32dd7'
	
	'''

	def __init__(
		self,
		exp	:	str	=	''
	)	->	None:
		self.exp = exp

	@property
	def exp(self):
		return self.exp
	
	@exp.setter
	def exp(self, new:str):
		self.exp = new

#for hover referencing
t.F_IDS()

def encode_C(
	instruction_set	:	list = []
)	->	str:
	
	#a single instruction comes in with a length of 6 in the format of:
	# [     int   ,[int T, int G], [<-ints] or int  , int  , int   , float]
	# [function ID, x source info, alpha source info, delta, delta2, kappa]

	#chromosome string representation
	C = ''
	#for each expression (gene) in the instruction set, condense into string
	for gene, i in enumerate(instruction_set):

		#sanity
		if(not isinstance(gene, list)):
			raise TypeError(f"Gene in instruction set should have came into encode_C as a list object. Got {type(gene)}")
		if(len(gene)<2):
			raise ValueError(f"Gene in instruction set came into encode_C with insufficient information. Got {gene}")
			
		#stack function declaration
		C+=f'F{gene[0]}'

		#stack source of x
		match(which_TG(gene)):
			case 0:
				C+=f'T{gene[1][0]}'

			case 1:
				C+=f'G{gene[1][1]}'
			
			case -1:
				raise ValueError(f"Gene# {i} in instruction set came into encode_C having improper x -> T/G selection. Got {gene}")

		#now onto variable declaration
		#have to check all implemented function cases frowny face

		#we will go alpha, delta, delta2, kappa

		#alpha case
		if(gene[0] in t.F_WITH('a')):
			#stack source of x
			match(which_TG(gene)):
				case 0:
					C+=f'aT{gene[2][0]}'

				case 1:
					C+=f'aG{gene[2][1]}'
				
				case -1:
					raise ValueError(f"Gene# {i} in instruction set came into encode_C having improper x -> T/G selection. Got {gene}")
		
		#delta case
		if(gene[0] in t.F_WITH('d')):
			C+=f'd{gene[3]}'
		
		#delta2 case
		if(gene[0] in t.f_WITH('dd')):
			C+=f'dd{gene[4]}'
		
		#kappa case
		if(gene[0] in t.F_WITH('k')):
			C+=f'k{round(gene[5], 3)}'

		#underscore to identify end of individual gene instructions
		C+='_'

	return C




def which_TG(
	gene	:	list
):
	'''returns the index of gene[1] for T or G selection. returns -1 if improperly formatted or illogical values'''
	if(len(gene)<2):
		raise ValueError(f"Gene in instruction set came into encode_C with insufficient information. Got {gene}")
	#passes if both are -1 or neither are -1
	if(~(gene[1][0]==-1 ^ gene[1][1]==-1)):
		return -1
	
	return 0 if (gene[1][0]==-1) else 1
	