import csv
import networkx as nx
import pandas as pd
import numpy as np


def create_graph(book_num):
    '''
    :param book_num: the book number from where we want to build up the graph
    :return: it returns the graph
    '''
    print('import data and creating graph from book' + str(book_num) + '...')
    input_file_handler = open("./dataset/book_" + str(book_num) + '.tsv', 'r', encoding="utf-8")  # open the handler
    csv_reader = csv.reader(input_file_handler, delimiter='\t', quotechar='"', quoting=csv.QUOTE_NONE)
    nodes_unweighted = list()  # list where store nodes
    for record in csv_reader:
        # print(record)
        node1 = record[0]  # pick starting node
        node2 = record[1]  # pick connected node

        nodes_unweighted.append((node1, node2))
        # nodes_unweighted.append((node1, node2, 1))  # append the tuple.

    input_file_handler.close()  # close the hendler

    # create the graph:
    graph = nx.Graph()

    graph.add_edges_from(nodes_unweighted)
    # graph.add_weighted_edges_from(nodes_unweighted)

    print('graph created!',
          'number of nodes in the graph: ' + str(graph.number_of_nodes()),
          'number of edges in the graph: ' + str(graph.number_of_edges()), '\n', sep='\n')
    return (graph)


def find_community(normalized_list):
    '''
    :param normalized_list: takes in input a list that contains the normalized PPR per each node.
    :return: it returns the best set of nodes with the associated lowest conductance
    '''
    candidate_set = set()  # is the candidate set to be the community
    complement_set = set(graph.nodes())  # is the complement set of nodes not included in the community
    best_sweep_index = -1  # sweep index used to selected the nodes that will belong to the community
    min_conductance = float("+inf")  # inizialite the best conductance to +inf
    for sweep in range(0, len(
            normalized_list) - 1):  # iterate over the length of the sorted node list according to their normalized score

        selected_node = normalized_list[sweep][0]  # pick the node
        candidate_set.add(selected_node)  # add it to the candidate set
        complement_set.remove(selected_node)  # remove it from the complement set
        conductance = nx.algorithms.cuts.conductance(graph,
                                                     candidate_set,
                                                     complement_set)  # compute the conductance

        # print('Conductance: '+str(conductance)+' and sweep: '+str(sweep))
        if conductance == 0. or conductance == 1.:  # skip conductance value of 0 and 1
            continue

        if conductance < min_conductance:  # if the given conductance is better than the one before then:
            min_conductance = conductance  # update the best min_conductance
            best_sweep_index = sweep  # update the best sweep index used to identify the nodes to put in the community

    best_set = set(
        [node_id for node_id, normalized_score in normalized_list[:best_sweep_index + 1]])  # return the community

    bestset_and_conductance = [best_set,
                               min_conductance]  # store a tuple with the community and the associated conductance value

    # print('Best conductance: '+str(min_conductance), 'Best Sweep: '+str(best_sweep_index),sep='\t\t')
    # pp.pprint(bestset_and_conductance)
    return (bestset_and_conductance)


def best_community(name, graph):
    """
    :param name: takes in input the character name used to find it's community

    It returns the community with the lowest conductance between all possibile combination of alpha and exponent

    :return: it returns a list as follow:
                - Best Community
                - Best Conductance
                - Best Alpha & Best Exponent combination that allowed to find the lowest Conductance
    """

    ##############################################################################################
    # (1) personalized dict that associates 1 if name == input character or 0 in all other cases #
    ##############################################################################################

    # In this part of the function, the personalization parameter is defined.
    # Basically it is defined the probability distribution of the teleporting among all nodes in the graph.

    personalized_dict = {node: (0 if node != name else 1) for node in graph.nodes()}
    # 0 is assigned if the node is not part of the topic.
    # 1 is assigned if the node is part of the topic,
    # (in this case we have only node because the PPR is composed by a topic with only 1 node.

    ######################################################
    # (2) compute the best pagerank per each alpha value #
    ######################################################

    # In this part of the function the PPR is computed per each alpha value.
    # As first step the PPR (based on a given alpha value is computed)
    # In section (2.1) the Normalized PPR is computed per each Exponent value.

    alphas = [round(n, 2) for n in np.arange(0.05, 1, 0.05)]
    best_alpha_conductance = float('+inf')  # variable that will contain the best alpha value
    best_alpha_set_conductance_exponent = list()  # will contain the best community with its associated conductance and the best combination of alpha and exponent
    for alpha_value in alphas:

        # compute the PPR according to alpha
        pagerank_personalized_result = nx.pagerank(graph,
                                                   alpha=alpha_value,
                                                   personalization=personalized_dict,
                                                   max_iter=100,
                                                   tol=1e-6
                                                   )

        #######################################################################################################
        # (2.1) compute the normalized personalized pagerank (PPR) changing exponent and return the best one. #
        #######################################################################################################

        # According to each exponent compute the Normalized PPR.
        # Per each Normalized PPR call the "find_community()" function in order to find out the set of nodes (community),
        # that have the lowest conductance according to the given combination of alpha and exponent.

        exponents = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        best_conductance = float('+inf')  # variable that will contain the overall best conductance.
        best_set_conductance_exponent = list()  # list that will contain the overall best community
        for exponent in exponents:
            ######print('EXPONENT = '+str(exponent))

            # compute the Normalized PPR and sort in discending order
            person_and_score_list = [(person, score / (graph.degree[person] ** exponent))
                                     for person, score in pagerank_personalized_result.items()]  # compute PPR
            person_and_score_list.sort(key=lambda x: (-x[1], x[0]))  # sort PPR in descending order

            # based on the given exponent, return the tuple (community, best conductance)

            ############################################################################
            # (2.2) call the "find_community()" function to return the community with  #
            # the lowest conductance according to the Normalized PPR                   #
            ############################################################################
            c_c_result = find_community(person_and_score_list)  # returns the comminity with the lowest conductance

            if c_c_result[1] < best_conductance:  # if its conductance is lower then the one already found store it
                best_conductance = c_c_result[1]  # update
                ######print('EXPONENT: '+str(exponent), 'CONDUCTANCE: '+str(c_c_result[1]),  'SET LENGTH: '+str(len(c_c_result[0])))
                best_set_conductance_exponent = [c_c_result[0], c_c_result[1], exponent]  # update best community

        print('ALPHA: ' + str(alpha_value),
              'EXPONENT: ' + str(best_set_conductance_exponent[2]),
              'CONDUCTANCE: ' + str(best_set_conductance_exponent[1]),
              sep='\t')

        if best_set_conductance_exponent[
            1] < best_alpha_conductance:  # if the conductance is better then the already found:
            best_alpha_conductance = best_set_conductance_exponent[1]  # update the best overall conductance
            best_alpha_set_conductance_exponent = [best_set_conductance_exponent[0],  # overall best community
                                                   best_set_conductance_exponent[1],  # overall best conductance value
                                                   best_set_conductance_exponent[2],  # best exponent value
                                                   alpha_value]  # best alpha value


    print(
        character + ' ==>',
        'BEST ALPHA: ' + str(best_alpha_set_conductance_exponent[3]),
        'BEST EXPONENT: ' + str(best_alpha_set_conductance_exponent[2]),
        'BEST CONDUCTANCE: ' + str(best_alpha_set_conductance_exponent[1]),
        'LENGTH: ' + str(len(best_alpha_set_conductance_exponent[0])),
        '\n',
        sep='\t'
    )
    return (best_alpha_set_conductance_exponent)


def familiy_counter(character_dict, book):
    """
    :param character_dict: a dictionary where:
                                - keys = are the characters
                                - values = is a list where:
                                            element_0 = best community,
                                            element_1 = best conductance,
                                            element_2 = exponent value,
                                            element_3 = alpha value
    :param book: the book number used to find the title of the book itself.
    :return: It returns a list of lists composed by 4 elemetents, each one representing a book.
             In each list there are 4 dictionary, one per each character.
             Every dictionary contains all the info required to fill the required output table.
    """

    book_list = list() # list that will contains results for every book

    for character, info in character_dict.items():

        books_names = ["A Game of Thrones", "A Clash of Kings",
                       "A Storm of Swords", "A Feast for Crows & A Dance with Dragons"]
        character_result = {
            'book_file_name': books_names[book - 1],
            'character_name': character,
            'conductance_value': info[1],
            'dumping_factor': info[3],
            'exponent': info[2],
            'tot_number_of_characters_in_community': len(info[0])
        }

        families = ["Baratheon", "Lannister", "Stark", "Targaryen"]
        for family in families:
            counter = 0 # counter used to count nodes belonging to a specific family
            for node in info[0]:
                if family.lower() in node.lower():
                    counter += 1
            character_result[family + '_family'] = counter # store the given number to the associated family

        book_list.append(character_result)
    print(book_list)
    return (book_list)

#####################################################################
# Start finding communities for every input character in every book #
#####################################################################

book_list = [n for n in range(1, 5)]

books_results = list()
for book in book_list:

    print('START PARSING BOOK_' + str(book))
    graph = create_graph(book)

    local_community_characters = ["Daenerys-Targaryen", "Jon-Snow", "Samwell-Tarly", "Tyrion-Lannister"]
    character_dict = dict()
    for character in local_community_characters:
        print('FINDING COMMUNITY FOR: ' + str(character) + '...')
        character_dict[character] = best_community(character, graph)

    books_results.append(familiy_counter(character_dict, book))
    print('\n\n\n')

books_df = pd.DataFrame([r for d in books_results for r in d]) # unpack the 4 lists into 1 list
books_df = books_df.sort_values(["book_file_name", "character_name"], ascending = True)

print(books_df.to_string())
print('\n', 'Writing .tsv file...')
books_df.to_csv('output.tsv', index=False, sep='\t')
print('Done!')






