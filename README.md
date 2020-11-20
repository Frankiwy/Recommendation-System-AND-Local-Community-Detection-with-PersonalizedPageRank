# Recommendation System & Local Community Detection with PersonalizedPageRank

![Image of Yaktocat](https://github.com/Frankiwy/Recommendation-System-AND-Local-Community-Detection-with-PersonalizedPageRank/blob/main/images/logo-sapienza-new.jpg)



### Part 1
In this part, you have to improve the performance of a recommendation-system by using non-trivial algorithms and also by performing the tuning of the hyper-parameters.

##### Part 1.1
Using the data available dataset folder, you must apply all algorithms for recommendation made available by “Surprise” libraries, according to their configuration.
For this part, and also for the next one, it is mandatory to use all CPU-cores available on your computer, by specifying the value in an explicit way.

##### Results Part 1.1
You have to “copy-paste” in the final report all the “TABLES” in output from the execution of the “cross_validate” command on all algorithms: the number of folds to use is equal to 5.
Moreover, you have to rank all recommendation algorithms you tested according to the MEAN_RMSE metric value: from the best to the worst algorithm.
Finally, you have to explain, by writing exactly one sentence, how you exploited all CPU-cores available on your machine.

##### Part 1.2
In this part, you have to improve the quality of both KNNBaseline and SVD algorithms, by performing hyper-parameters tuning always over five-folds. Even for this part, it is mandatory to use all CPU-cores available on your computer, and you have to use, again, the dataset available in dataset folder.
Only configurations with an average RMSE over all five folds less then 0.89 will be accepted. In particular, you have to perform a Random-Search-Cross-Validation process for tuning the hyper-parameter of the KNNBaseline algorithm. Instead, for tuning the hyper parameter of the SVD algorithm, you have to use a Grid-Search-Cross-Validation approach.


##### Results 1.2
By using at most two pages of the report, you must:
- put in the report the complete “Grid-of-Parameters” you used to increase the performances for each method.
- put in the report the best configuration you found for each method.
- put in the report the two average-RMSE associated to the two best estimators you tuned. 
- put in the report the total time required to select the best estimators.
- put in the report the number of CPU-cores you used.
- put in the report, by writing exactly one line, an explanation on how you exploited all CPU-cores available on your machine.

### PART 2
In this second part, it is requested to discover the social communities around particular characters of the well-known series of epic fantasy novels called “A Song of Ice and Fire”. 
Interactions among the characters of the novels are collected inside the four tsv files stored in the directory dataset. In particular, each file corresponds to a novel: "A Game of Thrones" (book_1.tsv), "A Clash of Kings" (book_2.tsv), and "A Storm of Swords" (book_3.tsv), "A Feast for Crows" merged with "A Dance with Dragons" (book_4.tsv)”. Each row in the .tsv files represents the fact that the names of the two characters represented in the first and second column appeared within 15 words of one another in the corresponding book.
What is requested it is to discover, for each book of the series, the local communities centered in only the following four characters: ”Daenerys-Targaryen”, ”Jon-Snow”, ”Samwell-Tarly” and ”Tyrion-Lannister”. For discovering these local communities you must create, for each provided book, an unweighted and undirected graph where nodes are characters of the book and where edges represent the interactions reported in the corresponding .tsv file.
The technique to use for discovering local communities has to consider the two following changes: 

1) Instead of using a single fixed value for the PageRank damping factor, you have to try all the following values: [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05].

2) Instead of normalizing the Personalized-PageRank value of each node in the graph by its degree as explained during the lectures, you have to implement the following more general normalization method: 

![\Medium NormalizedScore(v)=\frac{PPR(v)}{Degree(v)^{exponent}}](https://latex.codecogs.com/svg.latex?\Large&space;NormalizedScore(v)=\frac{PPR(v)}{Degree(v)^{exponent}})

Similarly to the previous point, for the ”exponent” variable you have to try all the following values: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0].

It is clear now that, for finding a local community with a good conductance value for a given character inside a particular book, you must run the modified local community detection method for each of the possible 19*6=114 configurations given by all combinations of the values of the following two parameters: “PageRank damping factor” and ”exponent”.

WARNING: Communities with a conductance value of 0 or 1 are not considered as valid communities.

It is important to remark that it is not requested to find a unique combination of parameters that is good for all inputs, but, what is requested, is to find a good ad-hoc combination of parameters for every single input.


#### Results for 2
By using at most two page of the report, you must represent as a table in the report the content of a tsv file with the following fields/columns:
- book_file_name.
- character_name.
- Dumping_factor_of_the_best_configuration.
- Exponent_of_the_best_configuration.
- Conductance_value_of_the_local_community.
- Number_of_Characters_inside_the_comunity_belonging_to_the_Baratheon_family.
- Number_of_Characters_inside_the_comunity_belonging_to_the_Lannister_family.
- Number_of_Characters_inside_the_comunity_belonging_to_the_Stark_family.
- Number_of_Characters_inside_the_comunity_belonging_to_the_Targaryen_family.
- Total_number_of_characters_inside_the_comunity

This .tsv file will contain 16 records/rows and must be sorted by ascending values of the first column and then by ascending values of the second column.
