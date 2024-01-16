import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import unicodedata
import pandas as pd
import json
import numpy as np 
import csv
import pickle


nombre_de_lignes = 0

with open('new_list_exp_reading.txt', 'r') as fichier:
    for ligne in fichier:
        nombre_de_lignes += 1

print(f"Nombre de lignes dans le fichier : {nombre_de_lignes}")



# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("dbddv01/gpt2-french-small")
model = AutoModelForCausalLM.from_pretrained("dbddv01/gpt2-french-small")
model.eval()

# Define hook functions
layer_outputs = []
def hook_layer_outputs(module, input, output):
    # Extract the primary activations from the output
    # Assuming the primary activations are the first element of the output tuple
    primary_activations = output[0] if isinstance(output, tuple) else output
    layer_outputs.append(primary_activations.squeeze().clone())

qkv = []
def hook_qkv(module, input, output):
    query, key, value = module.c_attn(input[0]).split(module.split_size, dim=2)
    qkv.append((query.squeeze().clone(), key.squeeze().clone(), value.squeeze().clone()))

# Register hooks
for layer in model.transformer.h:
    layer.register_forward_hook(hook_layer_outputs)
    layer.attn.register_forward_hook(hook_qkv)


print("bibi")

def fuse_tokens_words(tokens, stacked_vectors):
    word_vectors = []
    word_tokens = []
    temp_vectors = []
    temp_token = ""

    for token, vector in zip(tokens, stacked_vectors):
        if token.startswith("Ġ") or token.startswith("-") or token == ":":
            # Start a new word for 'Ġ', '-', or a colon
            if temp_vectors:
                word_vector = torch.cat(temp_vectors, dim=0)
                word_vectors.append(word_vector)
                word_tokens.append(temp_token)
                temp_vectors = []
            temp_token = token.lstrip("Ġ").lstrip("-").lstrip(":")
        else:
            # Append the token to the current word token
            temp_token += token.replace("Ġ", "")  # Removing 'Ġ' for continuation tokens
        temp_vectors.append(vector)

    # Add the last word
    if temp_vectors:
        word_vector = torch.cat(temp_vectors, dim=0)
        word_vectors.append(word_vector)
        word_tokens.append(temp_token)

    return word_tokens, word_vectors



def fix_encoding(token_list):
    fixed_tokens = []
    for token in token_list:
        try:
            # Attempt to fix the encoding
            fixed_token = token.encode('latin1').decode('utf8')
        except (UnicodeEncodeError, UnicodeDecodeError):
            # Use the original token if an error occurs
            fixed_token = token

        # Handling common encoding issues with French characters and other misencodings
        replacements = {
            "Ã©": "é",  # e acute
            "Ã¨": "è",  # e grave
            "Ã": "à",   # a grave
            "Ã´": "ô",  # o circumflex
            "Ãª": "ê",  # e circumflex
            "Ã¯": "ï",  # i diaeresis
            "Ã¼": "ü",  # u diaeresis
            "Ã§": "ç",  # c cedilla
            "â€™": "'",  # apostrophe
            "â€œ": "“",  # open quotation mark
            "â€": "”",  # close quotation mark
            "âĢĻ": "'",  # misencoded apostrophe or similar
            "àł": "à",   # Correct character for 'àł'
            "àª": "ê",   # Correct character for 'àª'
            "àī": "é",   # Correct character for 'àī'
            "Åĵ": "œ",   # Correct character for 'Åĵ'
            "Âł": " ",   # Correct character for 'Âł'
            # Add more replacements as needed
        }

        for wrong, correct in replacements.items():
            fixed_token = fixed_token.replace(wrong, correct)

        fixed_tokens.append(fixed_token)

    return fixed_tokens

sentences = ["La bise et le soleil. La bise et le soleil se disputaient, chacun assurant qu'il était le plus fort. Quand ils ont vu un voyageur qui s'avançait, enveloppé dans son manteau, ils sont tombés d'accord que celui qui arriverait le premier à le lui faire ôter, serait regardé comme le plus fort. Alors la bise s'est mise à souffler de toutes ses forces. Mais plus elle soufflait plus le voyageur serrait son manteau autour de lui. Finalement elle renonça à le lui faire ôter. Alors le soleil commença à briller. Et au bout d'un moment le voyageur réchauffé ôta son manteau. Ainsi la bise dû reconnaître que le soleil était le plus fort.", 
             "L’aveugle. Un aveugle avait l’habitude de reconnaître au toucher toute bête qu’on lui mettait entre les mains, et de dire de quelle espèce elle était. Or un jour on lui présenta un louveteau; il ne le parle pas et resta indécis. Je ne sais pas, dit-il, si c’est le petit d’un loup, d’un renard ou d’un autre animal du même genre; mais ce que je sais bien, c’est qu’il n’est pas fait pour aller avec un troupeau de moutons. C’est ainsi que le naturel des méchants se reconnaît souvent à l'extérieur.", 
             "Les dauphins, les baleines et le goujon. Des dauphins et des baleines se livraient bataille. Comme la lutte se prolongeait et devenait acharnée, un goujon c’est un petit poisson s’éleva à la surface et essaya de les réconcilier. Mais un des dauphins prenant la parole lui dit Il est moins humiliant pour nous de combattre et de périr les uns par les autres que de t’avoir pour médiateur. De même certains hommes qui n’ont aucune valeur, s’ils tombent sur un temps de troubles publics, s’imaginent qu’ils sont des personnages.", 
             "La mule. Une mule engraissée d’orge se mit à gambader, se disant à elle-même J’ai pour père un cheval rapide à la course, et moi je lui ressemble de tout point. Mais un jour l’occasion vint où la mule se vit forcée de courir. La course terminée, elle se renfrogna et se souvint soudain de son père l’âne. Cette fable montre que, même si les circonstances mettent un homme en vue, il ne doit pas oublier son origine; car cette vie n’est qu’incertitude.", 
             "La cigale et le renard. Une cigale chantait sur un arbre élevé. Un renard qui voulait la dévorer imagina la ruse que voici. Il se plaça en face d’elle, il admira sa belle voix et il l’invita à descendre il désirait, disait-il, voir l’animal qui avait une telle voix. Soupçonnant le piège, la cigale arracha une feuille et la laissa tomber. Le renard a couru, croyant que c’était la cigale. Tu te trompes, compère, lui dit-elle, si tu as cru que je descendrais. Je me défie des renards depuis le jour où j’ai vu dans la fiente de l’un d’eux des ailes de cigale. Les malheurs du voisin agissent les hommes sensés.", 
             "L’âne et le jardinier. Un âne était au service d’un jardinier. Comme il mangeait peu, tout en travaillant beaucoup, il pria Jupiter de le délivrer du jardinier et de le faire vendre à un autre maître. Zeus l’exauça et le fit vendre à un potier. Mais il fut de nouveau mécontent, parce qu’on le chargeait davantage et qu’on lui faisait porter l’argile et la poterie. Aussi demanda-t-il encore une fois à changer de maître, et il fut vendu à un corroyeur. Il tomba ainsi sur un maître pire que les autres. En voyant quel métier faisait ce maître, il dit en soupirant: Hélas ! malheureux que je suis ! j’aurais mieux fait de rester chez mes premiers maîtres; car celui-ci, à ce que je vois, tannera aussi ma peau. Cette fable montre que les serviteurs ne regrettent jamais tant leurs premiers maîtres que quand ils ont fait l’épreuve des suivants.", 
             "Les voyageurs et l’ours. Deux amis cheminaient sur la même route. Un ours leur apparut soudain. L'homme monta vite sur un arbre et s’y tint caché; l’autre, sur le point d’être pris, se laissa tomber sur le sol et contrefit le mort. L’ours approcha de lui son museau et le flaira partout; mais l’homme retenait sa respiration; car on dit que l’ours ne touche pas à un cadavre. Quand l’ours se fut éloigné, l’homme qui était sur l’arbre descendit et demanda à l’autre ce que l’ours lui avait dit à l’oreille. De ne plus voyager à l’avenir avec des amis qui se dérobent dans le danger, répondit l’autre. Cette fable montre que les amis véritables se reconnaissent à l’épreuve du malheur.", 
             "Le devin. Un devin, installé sur la place publique, y faisait recette. Soudain un quidam vint à lui et lui annonça que les portes de sa maison étaient ouvertes et qu’on avait enlevé tout ce qui était à l’intérieur. Hors de lui, il se leva d’un bond et courut en soupirant voir ce qui était arrivé. Un des gens qui se trouvaient là, le voyant courir, lui cria Hé ! l’ami, toi qui te piquais de prévoir ce qui doit arriver aux autres, tu n’as pas prévu ce qui t’arrive. On pourrait appliquer cette fable à ces gens qui règlent pitoyablement leur vie et qui se mêlent de diriger des affaires qui ne les regardent pas.", 
             "Une lampe enivrée d’huile, jetant une vive lumière, se vantait d’être plus brillante que le soleil. Mais un souffle de vent ayant sifflé, elle s’éteignit aussitôt. Quelqu’un la ralluma et lui dit Éclaire, lampe, et tais-toi. L’éclat des astres ne s’éclipse jamais. Il ne faut pas se laisser aveugler par l’orgueil, quand on est en réputation ou en honneur; car tout ce qui s’acquiert nous est étranger.", 
             "Le lion et le sanglier. Dans la saison d’été, quand la chaleur fait naître la soif, un lion et un sanglier varboirent à une petite source. Ils se querellèrent à qui boirait le premier, et de la querelle ils en vinrent à une lutte à mort. Mais soudain s’étant retournés pour reprendre haleine, ils virent des vautours qui attendaient pour dévorer celui qui tomberait le premier. Aussi, mettant fin à leur inimitié, ils dirent Il vaut mieux devenir amis que de servir de pâture à des vautours et à des corbeaux. Il est beau de mettre fin aux méchantes querelles et aux rivalités; car l’issue en est dangereuse pour tous les partis.", 
             "La royauté du lion. Un lion devint roi, qui n’était ni colère, ni cruel, ni violent, mais doux et juste, comme un homme. Il se fit sous son règne une assemblée générale des animaux, en vue de recevoir et de se donner mutuellement satisfaction, le loup au mouton, la panthère au chamois, le tigre au cerf, le chien au lièvre. Le lièvre peureux dit alors: J’ai vivement souhaité de voir ce jour, afin que les faibles paraissent redoutables aux violents. Quand la justice règne dans l’État, et que tous les jugements sont équitables, les humbles aussi vivent en tranquillité.", 
             "Les mouches. Du miel s’étant répandu dans un cellier, des mouches y volèrent et se mirent à le manger. C’était un régal si doux qu’elles ne pouvaient s’en détacher. Mais leurs pattes s’y étant engluées, elles ne purent prendre l’essor, et se sentant étouffer, elles dirent: Malheureuses que nous sommes, nous périssons pour un instant de plaisir. C’est ainsi que la gourmandise est souvent la cause de bien des maux.", 
             "La fourmi. La fourmi d’à présent était autrefois un homme qui, adonné à l’agriculture, ne se contentait pas du produit de ses propres travaux; il regardait d’un œil d’envie ceux des autres et ne cessait de dérober les fruits de ses voisins. Zeus indigné de sa cupidité le changea en l’animal que nous appelons fourmi. Mais pour avoir changé de forme, il n’a pas changé de caractère; car aujourd’hui encore il parcourt les champs, ramasse le blé et l’orge d’autrui, et les met en réserve"
             ]


word_dicts = []


for sentence in sentences:
    # Reset data captured by hooks
    layer_outputs.clear()
    qkv.clear()

    # Tokenize and process the sentence
    input_ids = tokenizer.encode(sentence, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    with torch.no_grad():
        model(input_ids)

    # Stack and then transpose the tensors
    stacked_layer_outputs = torch.stack(layer_outputs, dim=0)  # Stacks along a new dimension, resulting in [sequence_length, num_layers, feature_size, ...]
    transposed_layer_outputs = stacked_layer_outputs.transpose(0, 1)  # Transpose to get [num_layers, sequence_length, feature_size, ...]

    # Do the same for queries, keys, and values
    stacked_queries = torch.stack([q for q, _, _ in qkv], dim=0)
    transposed_queries = stacked_queries.transpose(0, 1)

    stacked_keys = torch.stack([k for _, k, _ in qkv], dim=0)
    transposed_keys = stacked_keys.transpose(0, 1)

    stacked_values = torch.stack([v for _, _, v in qkv], dim=0)
    transposed_values = stacked_values.transpose(0, 1)

    

    word_tokens, word_layer_outputs = fuse_tokens_words(tokens, transposed_layer_outputs)
    word_tokens, word_queries = fuse_tokens_words(tokens, transposed_queries)
    word_tokens, word_keys = fuse_tokens_words(tokens, transposed_keys)
    word_tokens, word_values = fuse_tokens_words(tokens, transposed_values)

    corrected_tokens = fix_encoding(word_tokens)

    # Create dictionaries for each word
    for word, lo, q, k, v in zip(corrected_tokens, word_layer_outputs, word_queries, word_keys, word_values):
        word_dict = {
            'word': word,
            'layer_outputs': lo.numpy(),
            'queries': q.numpy(),
            'keys': k.numpy(),
            'values': v.numpy()
        }
        word_dicts.append(word_dict)


print(word_dicts[12]['word'])

print(len(word_dicts))
words = [word_dict['word'] for word_dict in word_dicts]
with open("new_list_exp_reading_gpt.txt", 'w') as fichier:
    for element in words:
        fichier.write(element + '\n')




# Save using pickle
with open('word_dicts.pkl', 'wb') as f:
    pickle.dump(word_dicts, f)
