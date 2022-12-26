import json

def vcoco_index_2_cat(index): 
    index_2_cat = {0:'hold_obj',1:'stand',2:'sit_instr', 3:'ride_instr',4:'walk', 5:'look_obj',6:'hit_instr', 7:'hit_obj',8:'eat_obj', 9:'eat_instr',10:'jump_instr',11:'lay_instr',12:'talk_on_phone_instr',13:'carry_obj',14:'throw_obj',15:'catch_obj',16:'cut_instr',17:'cut_obj',18:'run',19:'work_on_computer_instr',20:'ski_instr',21:'surf_instr',22:'skateboard_instr',23:'smile',24:'drink_instr',25:'kick_obj',26:'point_instr',27:'read_obj',28:'snowboard_instr'}
    return index_2_cat[index]
    
def hico_index_2_cat(index): 
    index_2_cat =  {0:'adjust',1:'assemble',2:'block',3:'blow',4:'board',5:'break',6:'brush_with',7:'buy',8:'carry',9:'catch',10:'chase',11:'check',12:'clean',13:'control',14:'cook',15:'cut',16:'cut_with',    
17:'direct',18:'drag',  
19:'dribble',       
20:'drink_with',    
21:'drive',         
22:'dry',           
23:'eat',           
24:'eat_at',        
25:'exit',          
26:'feed',          
27:'fill',          
28:'flip',          
29:'flush',         
30:'fly',           
31:'greet',         
32:'grind',         
33:'groom',         
34:'herd',         
35:'hit',           
36:'hold',          
37:'hop_on',        
38:'hose',          
39:'hug',           
40:'hunt',          
41:'inspect',       
42:'install',       
43:'jump',          
44:'kick',          
45:'kiss',          
46:'lasso',         
47:'launch',        
48:'lick',          
49:'lie_on',        
50:'lift',          
51:'light',         
52:'load',          
53:'lose',          
54:'make',          
55:'milk',          
56:'move',          
57:'no_interaction',
58:'open',          
59:'operate',       
60:'pack',          
61:'paint',         
62:'park',          
63:'pay',           
64:'peel',          
65:'pet',           
66:'pick',          
67:'pick_up',       
68:'point',         
69:'pour',          
70:'pull',          
71:'push',          
72:'race',          
73:'read',          
74:'release',       
75:'repair',        
76:'ride',          
77:'row',           
78:'run',           
79:'sail',          
80:'scratch',       
81:'serve',         
82:'set',           
83:'shear',         
84:'sign',          
85:'sip',           
86:'sit_at',        
87:'sit_on',        
88:'slide',         
89:'smell',         
90:'spin',          
91:'squeeze',       
92:'stab',          
93:'stand_on',      
94:'stand_under',   
95:'stick',         
96:'stir',          
97:'stop_at',       
98:'straddle',      
99:'swing',         
100:'tag',           
101:'talk_on',       
102:'teach',         
103:'text_on',       
104:'throw',         
105:'tie',           
106:'toast',         
107:'train',         
108:'turn',          
109:'type_on',       
110:'walk',          
111:'wash',          
112:'watch',         
113:'wave',          
114:'wear',          
115:'wield',         
116:'zip'}
    return index_2_cat[index]

def vaw_index_2_cat(index): 
    anno_file = 'data/vaw/annotations/attribute_index.json'
    with open(anno_file, 'r') as f:
        annotations = json.load(f)
    index_2_cat = {v:k for k, v in annotations.items()}
    return index_2_cat[index]

def color_index():
    anno_file = 'attribute_index_color.json'
    with open(anno_file, 'r') as f:
        annotations = json.load(f)
    color_index = [v for k, v in annotations.items()]
    return color_index

