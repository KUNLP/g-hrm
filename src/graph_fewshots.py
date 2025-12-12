EXAMPLES = {
    "maple": """Definition of the graph: There are three types of nodes in the graph: paper, author and venue.
        Paper nodes have features: title, abstract, year and label. Author nodes have features: name. Venue nodes have features: name.
        Paper nodes are linked to their author nodes, venue nodes, reference paper nodes and cited_by paper nodes. Author nodes are linked to their paper nodes. Venue nodes are linked to their paper nodes.
        Question: When was the paper Strongly Interacting Higgs Sector in the Minimal Standard Model published?
        Thought 1: The question is asking some basic information of a node (Strongly Interacting Higgs Sector in the Minimal Standard Model). We need to find the node in the graph.
        Action 1: Retrieve[Strongly Interacting Higgs Sector in the Minimal Standard Model]
        Observation 1: The ID of this node is 3101448248.
        Thought 2: The question is asking the published date of a paper, we need to check the node feature (year) from the graph.
        Action 2: Feature[3101448248, year]
        Observation 2: 1993
        Thought 3: The published date of the paper is 1993.
        Action 3: Finish[1993]
        Question: When was the paper Strongly Interacting Higgs Sector in the Minimal Standard Model published?
        Thought 1: The question is asking some basic information of a node (Strongly Interacting Higgs Sector in the Minimal Standard Model). We need to find the node in the graph.
        Action 1: Retrieve[Strongly Interacting Higgs Sector in the Minimal Standard Model]
        Observation 1: The ID of this node is 3101448248.
        Thought 2: The question is asking the published date of a paper, we need to check the node feature (year) from the graph.
        Action 2: Feature[3101448248, year]
        Observation 2: 1993
        Thought 3: The published date of the paper is 1993.
        Action 3: Finish[1993]

        Definition of the graph: There are three types of nodes in the graph: paper, author and venue.
        Paper nodes have features: title, abstract, year and label. Author nodes have features: name. Venue nodes have features: name.
        Paper nodes are linked to their author nodes, venue nodes, reference paper nodes and cited_by paper nodes. Author nodes are linked to their paper nodes. Venue nodes are linked to their paper nodes.
        Question: How many authors do the paper Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars have?
        Thought 1: The question is asking information of a node (Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars). We need to find the node in the graph.
        Action 1: Retrieve[Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars]
        Observation 1: The ID of this node is 2090642949.
        Thought 2: The question is asking the number of authors of a paper, we need to calculate the node's author neighbor degree from the graph.
        Action 2: Degree[2090642949, author]
        Observation 2: 2
        Thought 3: The number of the authors is 2
        Action 3: Finish[2]

        Definition of the graph: There are three types of nodes in the graph: paper, author and venue.
        Paper nodes have features: title, abstract, year and label. Author nodes have features: name. Venue nodes have features: name.
        Paper nodes are linked to their author nodes, venue nodes, reference paper nodes and cited_by paper nodes. Author nodes are linked to their paper nodes. Venue nodes are linked to their paper nodes.
        Question: What was the publish venue of the paper Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars?
        Thought 1: The question is asking information of a node (Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars). We need to find the node in the graph.
        Action 1: Retrieve[Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars]
        Observation 1: The ID of this node is 2090642949.
        Thought 2: The question is asking the published venue of a paper, we need to check the node's venue neighbor from the graph.
        Action 2: Neighbor[2090642949, venue]
        Observation 2: ['1980519', '1053242']
        Thought 3: The ID of the published venue are 1980519 and 1053242. We need to get their names.
        Action 3: Feature[1980519, name], Feature[1053242, name]
        Observation 3: the astrophysical journal, the atmosphere journal
        Thought 4: The name of the published venues are the astrophysical journal and the atmosphere journal
        Action 4: Finish[the astrophysical journal, the atmosphere journal]
        """,
    "amazon": """Definition of the graph: There are two types of nodes in the graph: item and brand. 
            Item nodes have features: title, description, price, img, category. Brand nodes have features: name. 
            Item nodes are linked to their brand nodes, also_viewed_item nodes, buy_after_viewing_item nodes, also_bought_item nodes, bought_together_item nodes. Brand nodes are linked to their item nodes.
            Question: What is the price of the item Purple Sequin Tiny Dancer Tutu Ballet Dance Fairy Princess Costume Accessory?
            Thought 1: The question is asking some basic information of a item node (Purple Sequin Tiny Dancer Tutu Ballet Dance Fairy Princess Costume Accessory). We need to find the node in the graph.
            Action 1: Retrieve[Purple Sequin Tiny Dancer Tutu Ballet Dance Fairy Princess Costume Accessory]
            Observation 1: The ID of this node is 0000037214.
            Thought 2: The question is asking the price of a item, we need to check the node feature (price) from the graph.
            Action 2: Feature[0000037214, price]
            Observation 2: 6.99
            Thought 3: The price of the item is 6.99.
            Action 3: Finish[6.99]
            
            Definition of the graph: There are two types of nodes in the graph: item and brand. 
            Item nodes have features: title, description, price, img, category. Brand nodes have features: name. 
            Item nodes are linked to their brand nodes, also_viewed_item nodes, buy_after_viewing_item nodes, also_bought_item nodes, bought_together_item nodes. Brand nodes are linked to their item nodes.
            Question: How many also_viewed_item does item The Sherlock Holmes Audio Collection have?
            Thought 1: The question is asking information of a item node (The Sherlock Holmes Audio Collection). We need to find the node in the graph.
            Action 1: Retrieve[The Sherlock Holmes Audio Collection]
            Observation 1: The ID of this node is 0001048236.
            Thought 2: The question is asking the number of also_viewed_item of an item, we need to calculate the node's also_viewed_item neighbor degree from the graph.
            Action 2: Degree[0001048236, also_viewed_item]
            Observation 2: 9
            Thought 3: The number of also_viewed_item neighbors is 9.
            Action 3: Finish[9]
            
            Definition of the graph: There are two types of nodes in the graph: item and brand. 
            Item nodes have features: title, description, price, img, category. Brand nodes have features: name. 
            Item nodes are linked to their brand nodes, also_viewed_item nodes, buy_after_viewing_item nodes, also_bought_item nodes, bought_together_item nodes. Brand nodes are linked to their item nodes.
            Question: What is the brand of item Dr. Suess 19163 Dr. Seuss Puzzle 3 Pack Bundle?
            Thought 1: The question is asking information of a item node (Dr. Suess 19163 Dr. Seuss Puzzle 3 Pack Bundle). We need to find the node in the graph.
            Action 1: Retrieve[Dr. Suess 19163 Dr. Seuss Puzzle 3 Pack Bundle]
            Observation 1: The ID of this node is 0000191639.
            Thought 2: The question is asking the brand of a item, we need to check the node's brand neighbor from the graph.
            Action 2: Neighbor[0000191639, brand]
            Observation 2: ['brand_5']
            Thought 3: The brand of item is ['brand_5']. The ID of this brand node is brand_5. We need to check the brand node's feature (name) from the graph.
            Action 3: Feature[brand_5, name]
            Observation 3: Dr. Seuss.
            Thought 4: The brand name of the item is Dr. Seuss.
            Action 4: Finish[Dr. Seuss]
            """,
    "biomedical": """Definition of the graph: There are eleven types of nodes in the graph: Anatomy, Biological Process, Cellular Component, Compound, Disease, Gene, Molecular Function, Pathway, Pharmacologic Class, Side Effect, Symptom.
        Each node has name feature.
        There are these types of edges: Anatomy-downregulates-Gene, Anatomy-expresses-Gene, Anatomy-upregulates-Gene, Compound-binds-Gene, Compound-causes-Side Effect, Compound-downregulates-Gene, Compound-palliates-Disease, Compound-resembles-Compound, Compound-treats-Disease, Compound-upregulates-Gene, Disease-associates-Gene, Disease-downregulates-Gene, Disease-localizes-Anatomy, Disease-presents-Symptom, Disease-resembles-Disease, Disease-upregulates-Gene, Gene-covaries-Gene, Gene-interacts-Gene, Gene-participates-Biological Process, Gene-participates-Cellular Component, Gene-participates-Molecular Function, Gene-participates-Pathway, Gene-regulates-Gene, Pharmacologic Class-includes-Compound.
        Question: What compounds can be used to treat Crohn's disease? Please answer the compound names rather than IDs.
        Thought 1: The question is related to a disease node (Crohn's disease). We need to find the node in the graph.
        Action 1: Retrieve[Crohn's disease]
        Observation 1: The ID of this node is DOID:8778.
        Thought 2: The question is asking the compounds which can be used to treat a disease, we need to check the node's 'Compound-treats-Disease' neighbor from the graph.
        Action 2: Neighbor[DOID:8778, Compound-treats-Disease]
        Observation 2: ['DB01014', 'DB00244', 'DB00795', 'DB00993', 'DB00635', 'DB01033']
        Thought 3: The IDs of the compounds are 'DB01014', 'DB00244', 'DB00795', 'DB00993', 'DB00635', 'DB01033'. We need to get their names.
        Action 3: Feature[DB01014, name], Feature[DB00244, name], Feature[DB00795, name], Feature[DB00993, name], Feature[DB00635, name], Feature[DB01033, name]
        Observation 3: Balsalazide, Balsalazide, Mesalazine, Sulfasalazine, Azathioprine, Prednisone, Mercaptopurine
        Thought 4: The name of compounds are Balsalazide, Mesalazine, Sulfasalazine, Azathioprine, Prednisone, Mercaptopurine.
        Action 4: Finish[Balsalazide, Mesalazine, Sulfasalazine, Azathioprine, Prednisone, Mercaptopurine]
        
        Definition of the graph: There are eleven types of nodes in the graph: Anatomy, Biological Process, Cellular Component, Compound, Disease, Gene, Molecular Function, Pathway, Pharmacologic Class, Side Effect, Symptom.
        Each node has name feature.
        There are these types of edges: Anatomy-downregulates-Gene, Anatomy-expresses-Gene, Anatomy-upregulates-Gene, Compound-binds-Gene, Compound-causes-Side Effect, Compound-downregulates-Gene, Compound-palliates-Disease, Compound-resembles-Compound, Compound-treats-Disease, Compound-upregulates-Gene, Disease-associates-Gene, Disease-downregulates-Gene, Disease-localizes-Anatomy, Disease-presents-Symptom, Disease-resembles-Disease, Disease-upregulates-Gene, Gene-covaries-Gene, Gene-interacts-Gene, Gene-participates-Biological Process, Gene-participates-Cellular Component, Gene-participates-Molecular Function, Gene-participates-Pathway, Gene-regulates-Gene, Pharmacologic Class-includes-Compound.
        Question: What is the inchikey of Caffeine?
        Thought 1: The question is related to a compound node (Caffeine). We need to find the node in the graph.
        Action 1: Retrieve[Caffeine]
        Observation 1: The ID of this node is DB00201.
        Thought 2: The question is asking the inchikey feature of a node, we need to check the node's 'inchikey' feature from the graph.
        Action 2: Feature[DB00201, inchikey]
        Observation 2: InChIKey=RYYVLZVUVIJVGH-UHFFFAOYSA-N
        Thought 3: The inchikey of the node is InChIKey=RYYVLZVUVIJVGH-UHFFFAOYSA-N.
        Action 3: Finish[InChIKey=RYYVLZVUVIJVGH-UHFFFAOYSA-N]
        
        Definition of the graph: There are eleven types of nodes in the graph: Anatomy, Biological Process, Cellular Component, Compound, Disease, Gene, Molecular Function, Pathway, Pharmacologic Class, Side Effect, Symptom.
        Each node has name feature.
        There are these types of edges: Anatomy-downregulates-Gene, Anatomy-expresses-Gene, Anatomy-upregulates-Gene, Compound-binds-Gene, Compound-causes-Side Effect, Compound-downregulates-Gene, Compound-palliates-Disease, Compound-resembles-Compound, Compound-treats-Disease, Compound-upregulates-Gene, Disease-associates-Gene, Disease-downregulates-Gene, Disease-localizes-Anatomy, Disease-presents-Symptom, Disease-resembles-Disease, Disease-upregulates-Gene, Gene-covaries-Gene, Gene-interacts-Gene, Gene-participates-Biological Process, Gene-participates-Cellular Component, Gene-participates-Molecular Function, Gene-participates-Pathway, Gene-regulates-Gene, Pharmacologic Class-includes-Compound.
        Question: How many side effects does Caffeine have?
        Thought 1: The question is related to a compound node (Caffeine). We need to find the node in the graph.
        Action 1: Retrieve[Caffeine]
        Observation 1: The ID of this node is DB00201.
        Thought 2: The question is asking the number of side effects a compound has, we need to calculate the number of the node's 'Compound-causes-Side Effect' neighbors from the graph.
        Action 2: Degree[DB00201, 'Compound-causes-Side Effect']
        Observation 2: 58
        Thought 3: The number of 'Compound-causes-Side Effect' neighbors are 58.
        Action 3: Finish[58]
            """,
    "legal": """Definition of the graph: There are four types of nodes in the graph: opinion, opinion_cluster, docket, and court.
        Opinion nodes have features: plain_text. Opinion_cluster nodes have features: syllabus, judges, case_name, attorneys. Docket nodes have features: pacer_case_id, case_name. Court nodes have features: full_name, start_date, end_date, citation_string.
        Opinion nodes are linked to their reference nodes and cited_by nodes, as well as their opinion_cluster nodes. Opinion_cluster nodes are linked to opinion nodes and docket nodes. Docket nodes are linked to opinion_cluster nodes and court nodes. Court nodes are linked to docket nodes.
        Question: Who are the judges that involved in the opinion_cluster: <p>Constitutional law — Title of act — Mortgage—Deed—Liability of grantee — Act of June IB, 1878, P. L. BOB.</p> <p>1. The Act of June 12, 1878, P. L. 205, entitled “An Act to define the liability of grantees of real estate for the incumbrances thereon,” is sufficiently comprehensive in its title to cover both the first and second sections of the act, and does not violate art. Ill, sec. 3 of the constitution of Pennsylvania.</p> <p>2. Where an owner of land creates two mortgages thereon and then conveys the property subject to the mortgages, the grantee covenanting in the deed itself to assume and pay the mortgages as a part of the consideration, and thereafter the grantee conveys the land to another person without expressly assuming any continuing liability to pay the mortgages, such grantee is not liable to the mortgag^ for payment of the mortgages. £V</p>?
        Thought 1: The question is related to a opinion_cluster (<p>Constitutional law — Title of act — Mortgage—Deed—Liability of grantee — Act of June IB, 1878, P. L. BOB.</p> <p>1. The Act of June 12, 1878, P. L. 205, entitled “An Act to define the liability of grantees of real estate for the incumbrances thereon,” is sufficiently comprehensive in its title to cover both the first and second sections of the act, and does not violate art. Ill, sec. 3 of the constitution of Pennsylvania.</p> <p>2. Where an owner of land creates two mortgages thereon and then conveys the property subject to the mortgages, the grantee covenanting in the deed itself to assume and pay the mortgages as a part of the consideration, and thereafter the grantee conveys the land to another person without expressly assuming any continuing liability to pay the mortgages, such grantee is not liable to the mortgag^ for payment of the mortgages. £V</p>). We need to find the node in the graph.
        Action 1: Retrieve[<p>Constitutional law — Title of act — Mortgage—Deed—Liability of grantee — Act of June IB, 1878, P. L. BOB.</p> <p>1. The Act of June 12, 1878, P. L. 205, entitled “An Act to define the liability of grantees of real estate for the incumbrances thereon,” is sufficiently comprehensive in its title to cover both the first and second sections of the act, and does not violate art. Ill, sec. 3 of the constitution of Pennsylvania.</p> <p>2. Where an owner of land creates two mortgages thereon and then conveys the property subject to the mortgages, the grantee covenanting in the deed itself to assume and pay the mortgages as a part of the consideration, and thereafter the grantee conveys the land to another person without expressly assuming any continuing liability to pay the mortgages, such grantee is not liable to the mortgag^ for payment of the mortgages. £V</p>]
        Observation 1: The ID of this node is opc-6381448.
        Thought 2: The question is asking the judges of this opinion_cluster node, we need to check the node's 'judges' feature from the graph.
        Action 2: Feature[opc-6381448, judges]
        Observation 2: Brown, Elkin, Fell, Mestrezat, Moschzisker, Potter, Stewart
        Thought 3: The judges of this opinion cluster is Brown, Elkin, Fell, Mestrezat, Moschzisker, Potter, Stewart
        Action 3: Finish[Brown, Elkin, Fell, Mestrezat, Moschzisker, Potter, Stewart]

        Definition of the graph: There are four types of nodes in the graph: opinion, opinion_cluster, docket, and court.
        Opinion nodes have features: plain_text. Opinion_cluster nodes have features: syllabus, judges, case_name, attorneys. Docket nodes have features: pacer_case_id, case_name. Court nodes have features: full_name, start_date, end_date, citation_string.
        Opinion nodes are linked to their reference nodes and cited_by nodes, as well as their opinion_cluster nodes. Opinion_cluster nodes are linked to opinion nodes and docket nodes. Docket nodes are linked to opinion_cluster nodes and court nodes. Court nodes are linked to docket nodes.
        Question: How many opinions are contained in this opinion_cluster: <p>A general statement that the decision of a state court is against the constitutional rights of the objecting party, or against the Fourteenth Amendment, or that it is without due process of law,-particularly when these objections appear only in specifications of error, so called, will not raise a Federal question, even where the judgment is a final one within Rev. Stat. § 709.</p> <p>In these cases there was no final judgment, such as is provided for in Rev. Stat. § 709, and there does not appear to have arisen any Federal question whatever.</p>'},
        Thought 1: The question is related to a opinion_cluster (<p>A general statement that the decision of a state court is against the constitutional rights of the objecting party, or against the Fourteenth Amendment, or that it is without due process of law,-particularly when these objections appear only in specifications of error, so called, will not raise a Federal question, even where the judgment is a final one within Rev. Stat. § 709.</p> <p>In these cases there was no final judgment, such as is provided for in Rev. Stat. § 709, and there does not appear to have arisen any Federal question whatever.</p>). We need to find the node in the graph.
        Action 1: Retrieve[<p>A general statement that the decision of a state court is against the constitutional rights of the objecting party, or against the Fourteenth Amendment, or that it is without due process of law,-particularly when these objections appear only in specifications of error, so called, will not raise a Federal question, even where the judgment is a final one within Rev. Stat. § 709.</p> <p>In these cases there was no final judgment, such as is provided for in Rev. Stat. § 709, and there does not appear to have arisen any Federal question whatever.</p>]
        Observation 1: The ID of this node is opc-94592.
        Thought 2: The question is asking the number of the opinion neighbors of this node, we need to calculate the node's "opinion" neighbor degree from the graph.
        Action 2: Degree[opc-94592, opinion]
        Observation 2: 1
        Thought 3: The number of the opinion neighbors is 1
        Action 3: Finish[1]

        Definition of the graph: There are four types of nodes in the graph: opinion, opinion_cluster, docket, and court.
        Opinion nodes have features: plain_text. Opinion_cluster nodes have features: syllabus, judges, case_name, attorneys. Docket nodes have features: pacer_case_id, case_name. Court nodes have features: full_name, start_date, end_date, citation_string.
        Opinion nodes are linked to their reference nodes and cited_by nodes, as well as their opinion_cluster nodes. Opinion_cluster nodes are linked to opinion nodes and docket nodes. Docket nodes are linked to opinion_cluster nodes and court nodes. Court nodes are linked to docket nodes.
        Question: What is the docket pacer case ID for this opinion cluster: In May, 1873, a brewer pays to the collector $100 as special tax, for which a special-tax stamp is issued to Mm. Subsequently he shoivs to. the satisfaction of the Commissions)\' of Internal Revenue that he was not a brewer of more than 500 barrels, and henee that he is entitled to have $50 refunded. The Commissioner issues his certificate for a special-tax stamp of the value of%50 returned. The Treasury refuses payment of the certificate. The breioe)\' brings his action to recover the amount thereof.</p> <p>I. The Commissioner of Internal Revenue being authorized by one section of the statutes (Rev. Stat., § 3220) to refund taxes “unjustly assessed or excessive in, amount," and by another, (§ 3426,) the value of special-tax stamps ‘‘where the rates of duties represented thereby have been paid in error or remitted," it is immaterial whether his certificate be in form under the one section or the other so long as it represents a case where the Commissioner is authorized to refund and the amoirnt which the party is entitled to have refunded.</p> <p>II. When a controversy under the revenue laws is by law required to be determined by other officers and other tribunals, this court is without jurisdiction; but where the proper officer has determined a question, and awarded an allowance, and filed a certificate, and exhausted his jurisdiction, and the Treasury has refused to carry out the award by mating payment, this court has jurisdiction of an action brought to recover the. award..
        Thought 1: The question is related to a opinion cluster (In May, 1873, a brewer pays to the collector $100 as special tax, for which a special-tax stamp is issued to Mm. Subsequently he shoivs to. the satisfaction of the Commissions)\' of Internal Revenue that he was not a brewer of more than 500 barrels, and henee that he is entitled to have $50 refunded. The Commissioner issues his certificate for a special-tax stamp of the value of%50 returned. The Treasury refuses payment of the certificate. The breioe)\' brings his action to recover the amount thereof.</p> <p>I. The Commissioner of Internal Revenue being authorized by one section of the statutes (Rev. Stat., § 3220) to refund taxes “unjustly assessed or excessive in, amount," and by another, (§ 3426,) the value of special-tax stamps ‘‘where the rates of duties represented thereby have been paid in error or remitted," it is immaterial whether his certificate be in form under the one section or the other so long as it represents a case where the Commissioner is authorized to refund and the amoirnt which the party is entitled to have refunded.</p> <p>II. When a controversy under the revenue laws is by law required to be determined by other officers and other tribunals, this court is without jurisdiction; but where the proper officer has determined a question, and awarded an allowance, and filed a certificate, and exhausted his jurisdiction, and the Treasury has refused to carry out the award by mating payment, this court has jurisdiction of an action brought to recover the. award.). We need to find the node in the graph.
        Action 1: Retrieve[In May, 1873, a brewer pays to the collector $100 as special tax, for which a special-tax stamp is issued to Mm. Subsequently he shoivs to. the satisfaction of the Commissions)\' of Internal Revenue that he was not a brewer of more than 500 barrels, and henee that he is entitled to have $50 refunded. The Commissioner issues his certificate for a special-tax stamp of the value of%50 returned. The Treasury refuses payment of the certificate. The breioe)\' brings his action to recover the amount thereof.</p> <p>I. The Commissioner of Internal Revenue being authorized by one section of the statutes (Rev. Stat., § 3220) to refund taxes “unjustly assessed or excessive in, amount," and by another, (§ 3426,) the value of special-tax stamps ‘‘where the rates of duties represented thereby have been paid in error or remitted," it is immaterial whether his certificate be in form under the one section or the other so long as it represents a case where the Commissioner is authorized to refund and the amoirnt which the party is entitled to have refunded.</p> <p>II. When a controversy under the revenue laws is by law required to be determined by other officers and other tribunals, this court is without jurisdiction; but where the proper officer has determined a question, and awarded an allowance, and filed a certificate, and exhausted his jurisdiction, and the Treasury has refused to carry out the award by mating payment, this court has jurisdiction of an action brought to recover the. award.]
        Observation 1: The ID of this node is opc-8599951.
        Thought 2: The question is asking the judges of this opinion cluster node, we need to check the node's 'judges' feature from the graph.
        Action 2: Feature[opc-8599951]
        Observation 2: {'judges': 'Eichaedson', 'case_name': 'Kaufman v. United States', 'attorneys': 'Mr. P. P. Pye for the claimant., Mr. Assistant Attorney-General Simons for the defendants.', 'syllabus': '<p>In May, 1873, a brewer pays to the collector $100 as special tax, for which a special-tax stamp is issued to Mm. Subsequently he shoivs to. the satisfaction of the Commissions)\' of Internal Revenue that he was not a brewer of more than 500 barrels, and henee that he is entitled to have $50 refunded. The Commissioner issues his certificate for a special-tax stamp of the value of%50 returned. The Treasury refuses payment of the certificate. The breioe)\' brings his action to recover the amount thereof.</p> <p>I. The Commissioner of Internal Revenue being authorized by one section of the statutes (Rev. Stat., § 3220) to refund taxes “unjustly assessed or excessive in, amount," and by another, (§ 3426,) the value of special-tax stamps ‘‘where the rates of duties represented thereby have been paid in error or remitted," it is immaterial whether his certificate be in form under the one section or the other so long as it represents a case where the Commissioner is authorized to refund and the amoirnt which the party is entitled to have refunded.</p> <p>II. When a controversy under the revenue laws is by law required to be determined by other officers and other tribunals, this court is without jurisdiction; but where the proper officer has determined a question, and awarded an allowance, and filed a certificate, and exhausted his jurisdiction, and the Treasury has refused to carry out the award by mating payment, this court has jurisdiction of an action brought to recover the. award.</p>'}
        Thought 3: The judges of this opinion cluster is Eichaedson
        Action 3: Finish[Eichaedson]
        """,
    "goodreads": """Definition of the graph: There are four types of nodes in the graph: book, author, publisher, and series. 
        Book nodes have features: country_code, language_code, is_ebook, title, description, format, num_pages, publication_year, url, popular_shelves, and genres. Author nodes have features: name. Publisher nodes have features: name. Series nodes have features: title and description. 
        Book nodes are linked to their author nodes, publisher nodes, series nodes and similar_books nodes. Author nodes are linked to their book nodes. Publisher nodes are linked to their book nodes. Series nodes are linked to their book nodes.
        Question: Who is the author of the book 'The Great Gatsby'?
        Thought 1: The question is asking for the author of a specific book (The Great Gatsby). We need to find the book node in the graph.
        Action 1: Retrieve[The Great Gatsby]
        Observation 1: The ID of this book node is 21366268.
        Thought 2: The question is asking for the author of the book. We need to check the book node's author neighbor from the graph.
        Action 2: Neighbor[21366268, author]
        Observation 2: ['3190']
        Thought 3: The ID of the author is 3190. We need to check the name.
        Action 3: Feature[3190, name]
        Observation 3: F. Scott Fitzgerald
        Thought 4: The author of 'The Great Gatsby' is F. Scott Fitzgerald.
        Action 4: Finish[F. Scott Fitzgerald]
        
        Definition of the graph: There are four types of nodes in the graph: book, author, publisher, and series. 
        Book nodes have features: country_code, language_code, is_ebook, title, description, format, num_pages, publication_year, url, popular_shelves, and genres. Author nodes have features: name. Publisher nodes have features: name. Series nodes have features: title and description. 
        Book nodes are linked to their author nodes, publisher nodes, series nodes and similar_books nodes. Author nodes are linked to their book nodes. Publisher nodes are linked to their book nodes. Series nodes are linked to their book nodes.
        Question: What is the genre of the book '1984'?
        Thought 1: The question is asking for the genre of a specific book (1984). We need to find the book node in the graph.
        Action 1: Retrieve[1984]
        Observation 1: The ID of this book node is 3475269.
        Thought 2: The question is asking for the genre of the book. We need to check the book node's genre feature from the graph.
        Action 2: Feature[3475269, genres]
        Observation 2: currently-reading, classics, fiction
        Thought 3: The genres of the book '1984' are currently-reading, classics, fiction.
        Action 3: Finish[currently-reading, classics, fiction]
        
        Definition of the graph: There are four types of nodes in the graph: book, author, publisher, and series. 
        Book nodes have features: country_code, language_code, is_ebook, title, description, format, num_pages, publication_year, url, popular_shelves, and genres. Author nodes have features: name. Publisher nodes have features: name. Series nodes have features: title and description. 
        Book nodes are linked to their author nodes, publisher nodes, series nodes and similar_books nodes. Author nodes are linked to their book nodes. Publisher nodes are linked to their book nodes. Series nodes are linked to their book nodes.
        Question: How many series included the book 'Harry Potter and the Sorcerer's Stone'?
        Thought 1: We need to find the book node for 'Harry Potter and the Sorcerer's Stone' to determine its series.
        Action 1: Retrieve[Harry Potter and the Sorcerer's Stone]
        Observation 1: The ID of this book node is 27421523.
        Thought 2: Now that we have the book node, we need to calculate the node's series neighbor degree from the graph.
        Action 2: Degree[27421523, series]
        Observation 2: 1
        Thought 3: The number of series neighbors is 1.
        Action 3: Finish[1]
        """,
    "dblp": """Definition of the graph: There are three types of nodes in the graph: paper, author and venue.
        Paper nodes have features: title, abstract, keywords, lang, and year. Author nodes have features: name and organization. Venue nodes have features: name.
        Paper nodes are linked to their author nodes, venue nodes, reference nodes (the papers this paper cite) and cited_by nodes (other papers which cite this paper). Author nodes are linked to their paper nodes. Venue nodes are linked to their paper nodes.
        Question: What organization is researcher Greg Daville from?
        Thought 1: The question is asking about the organization of a researcher named Greg Daville. We need to find the node corresponding to Greg Daville in the graph.
        Action 1: Retrieve[Greg Daville]
        Observation 1: The ID of this retrieval target node is 53f460a7dabfaee4dc83702a.
        Thought 2: The question is asking for the organization of a researcher, so we need to check the node's organization feature from the graph.
        Action 2: Feature[53f460a7dabfaee4dc83702a, organization]
        Observation 2: Hove East Sussex, United Kingdom 
        Thought 3: The organization of researcher Greg Daville is Hove East Sussex, United Kingdom.
        Action 3: Finish[Hove East Sussex, United Kingdom] 
    
        Definition of the graph: There are three types of nodes in the graph: paper, author and venue.
        Paper nodes have features: title, abstract, keywords, lang, and year. Author nodes have features: name and organization. Venue nodes have features: name.
        Paper nodes are linked to their author nodes, venue nodes, reference nodes (the papers this paper cite) and cited_by nodes (other papers which cite this paper). Author nodes are linked to their paper nodes. Venue nodes are linked to their paper nodes.
        Question: How many papers are written by author Nicholas Lydon?
        Thought 1: The question is asking for the number of written papers of a specific author (Nicholas Lydon). We need to find the author node in the graph.
        Action 1: Retrieve[Nicholas Lydon]
        Observation 1: The ID of this retrieval target node is 53f438c3dabfaedf43596117.
        Thought 2: The question is asking for the number of papers written by Nicholas Lydon. We need to calculate the "paper" neighbor degree of this node.
        Action 2: Degree[53f438c3dabfaedf43596117, paper]
        Observation 2: 2
        Thought 3: The number of the paper neighbors is 2
        Action 3: Finish[2]        

        Definition of the graph: There are three types of nodes in the graph: paper, author and venue.
        Paper nodes have features: title, abstract, keywords, lang, and year. Author nodes have features: name and organization. Venue nodes have features: name.
        Paper nodes are linked to their author nodes, venue nodes, reference nodes (the papers this paper cite) and cited_by nodes (other papers which cite this paper). Author nodes are linked to their paper nodes. Venue nodes are linked to their paper nodes.
        Question: What was the publish venue of the paper Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars?
        Thought 1: The question is asking information of a node (Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars). We need to find the node in the graph.
        Action 1: Retrieve[Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars]
        Observation 1: The ID of this node is 2090642949.
        Thought 2: The question is asking the published venue of a paper, we need to check the node's venue neighbor from the graph.
        Action 2: Neighbor[2090642949, venue]
        Observation 2: ['1980519', '1053242']
        Thought 3: The ID of the published venue are 1980519 and 1053242. We need to get their names.
        Action 3: Feature[1980519, name], Feature[1053242, name]
        Observation 3: the astrophysical journal, the atmosphere journal
        Thought 4: The name of the published venues are the astrophysical journal and the atmosphere journal
        Action 4: Finish[the astrophysical journal, the atmosphere journal]
        """,
}

SHORT_EXAMPLES = {
    "maple": """Definition of the graph: There are three types of nodes in the graph: paper, author and venue.
        Paper nodes have features: title, abstract, year and label. Author nodes have features: name. Venue nodes have features: name.
        Paper nodes are linked to their author nodes, venue nodes, reference paper nodes and cited_by paper nodes. Author nodes are linked to their paper nodes. Venue nodes are linked to their paper nodes.
        Question: When was the paper Strongly Interacting Higgs Sector in the Minimal Standard Model published?
        Thought 1: The question is asking some basic information of a node (Strongly Interacting Higgs Sector in the Minimal Standard Model). We need to find the node in the graph and check the node feature (year) from the graph.
        Action 1: Feature[Retrieve[Strongly Interacting Higgs Sector in the Minimal Standard Model], year]
        Observation 1: The ID of this node is 3101448248. The year is 1993.
        Thought 2: The published year of this paper is 1993.
        Action 2: Finish[1993]

        Definition of the graph: There are three types of nodes in the graph: paper, author and venue.
        Paper nodes have features: title, abstract, year and label. Author nodes have features: name. Venue nodes have features: name.
        Paper nodes are linked to their author nodes, venue nodes, reference paper nodes and cited_by paper nodes. Author nodes are linked to their paper nodes. Venue nodes are linked to their paper nodes.
        Question: How many authors do the paper Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars have?
        Thought 1: The question is asking information of a node (Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars). We need to find the node in the graph.
        Action 1: Degree[Retrieve[Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars], author]
        Observation 1: The ID of this node is 2090642949. Degree is 2.
        Thought 2: The number of the authors is 2.
        Action 2: Finish[2]

        Definition of the graph: There are three types of nodes in the graph: paper, author and venue.
        Paper nodes have features: title, abstract, year and label. Author nodes have features: name. Venue nodes have features: name.
        Paper nodes are linked to their author nodes, venue nodes, reference paper nodes and cited_by paper nodes. Author nodes are linked to their paper nodes. Venue nodes are linked to their paper nodes.
        Question: What was the publish venue of the paper Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars?
        Thought 1: The question is asking information of a node (Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars). We need to find the node in the graph.
        Action 1: Retrieve[Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars]
        Observation 1: The ID of this node is 2090642949.
        Thought 2: The question is asking the published venue of a paper, we need to check the node's venue neighbor from the graph and get their names.
        Action 2: Feature[Neighbor[2090642949, venue], name]
        Observation 2: ['1980519', '1053242'], the astrophysical journal, the atmosphere journal
        Thought 3: The ID of the published venue are 1980519 and 1053242. The name of the published venues are the astrophysical journal and the atmosphere journal.
        Action 3: Finish[the astrophysical journal, the atmosphere journal]

        Definition of the graph: There are three types of nodes in the graph: paper, author and venue.
        Paper nodes have features: title, abstract, year and label. Author nodes have features: name. Venue nodes have features: name.
        Paper nodes are linked to their author nodes, venue nodes, reference paper nodes and cited_by paper nodes. Author nodes are linked to their paper nodes. Venue nodes are linked to their paper nodes.
        Question: When was the paper Strongly Interacting Higgs Sector in the Minimal Standard Model published?
        Thought 1: The question is asking some basic information of a node (Strongly Interacting Higgs Sector in the Minimal Standard Model). We need to find the node in the graph.
        Action 1: Retrieve[Strongly Interacting Higgs Sector in the Minimal Standard Model]
        Observation 1: The ID of this node is 3101448248.
        Thought 2: The question is asking the published date of a paper, we need to check the node feature (year) from the graph.
        Action 2: Feature[3101448248, year]
        Observation 2: 1993
        Thought 3: The published date of the paper is 1993.
        Action 3: Finish[1993]

        Definition of the graph: There are three types of nodes in the graph: paper, author and venue.
        Paper nodes have features: title, abstract, year and label. Author nodes have features: name. Venue nodes have features: name.
        Paper nodes are linked to their author nodes, venue nodes, reference paper nodes and cited_by paper nodes. Author nodes are linked to their paper nodes. Venue nodes are linked to their paper nodes.
        Question: How many authors do the paper Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars have?
        Thought 1: The question is asking information of a node (Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars). We need to find the node in the graph.
        Action 1: Retrieve[Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars]
        Observation 1: The ID of this node is 2090642949.
        Thought 2: The question is asking the number of authors of a paper, we need to calculate the node's author neighbor degree from the graph.
        Action 2: Degree[2090642949, author]
        Observation 2: 2
        Thought 3: The number of the authors is 2
        Action 3: Finish[2]

        Definition of the graph: There are three types of nodes in the graph: paper, author and venue.
        Paper nodes have features: title, abstract, year and label. Author nodes have features: name. Venue nodes have features: name.
        Paper nodes are linked to their author nodes, venue nodes, reference paper nodes and cited_by paper nodes. Author nodes are linked to their paper nodes. Venue nodes are linked to their paper nodes.
        Question: What was the publish venue of the paper Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars?
        Thought 1: The question is asking information of a node (Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars). We need to find the node in the graph.
        Action 1: Retrieve[Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars]
        Observation 1: The ID of this node is 2090642949.
        Thought 2: The question is asking the published venue of a paper, we need to check the node's venue neighbor from the graph.
        Action 2: Neighbor[2090642949, venue]
        Observation 2: ['1980519', '1053242']
        Thought 3: The ID of the published venue are 1980519 and 1053242. We need to get their names.
        Action 3: Feature[1980519, name], Feature[1053242, name]
        Observation 3: the astrophysical journal, the atmosphere journal
        Thought 4: The name of the published venues are the astrophysical journal and the atmosphere journal
        Action 4: Finish[the astrophysical journal, the atmosphere journal]
        """,
        "amazon": """Definition of the graph: There are two types of nodes in the graph: item and brand. 
            Item nodes have features: title, description, price, img, category. Brand nodes have features: name. 
            Item nodes are linked to their brand nodes, also_viewed_item nodes, buy_after_viewing_item nodes, also_bought_item nodes, bought_together_item nodes. Brand nodes are linked to their item nodes.
            Question: What is the price of the item Purple Sequin Tiny Dancer Tutu Ballet Dance Fairy Princess Costume Accessory?
            Thought 1: The question is asking the price of a item node (Purple Sequin Tiny Dancer Tutu Ballet Dance Fairy Princess Costume Accessory). We need to find the node in the graph and check the node feature (price) from the graph.
            Action 1: Feature[Retrieve[Purple Sequin Tiny Dancer Tutu Ballet Dance Fairy Princess Costume Accessory], price]
            Observation 1: The ID of this node is 0000037214. The price of the item is 6.99
            Thought 2: The price of the item is 6.99.
            Action 2: Finish[6.99]
            
            Definition of the graph: There are two types of nodes in the graph: item and brand. 
            Item nodes have features: title, description, price, img, category. Brand nodes have features: name. 
            Item nodes are linked to their brand nodes, also_viewed_item nodes, buy_after_viewing_item nodes, also_bought_item nodes, bought_together_item nodes. Brand nodes are linked to their item nodes.
            Question: How many also_viewed_item does item The Sherlock Holmes Audio Collection have?
            Thought 1: The question is asking information of a item node (The Sherlock Holmes Audio Collection). We need to find the node in the graph and calculate the node's also_viewed_item neighbor degree from the graph.
            Action 1: Degree[Retrieve[The Sherlock Holmes Audio Collection], also_viewed_item]
            Observation 1: The ID of this node is 0001048236. 9
            Thought 2: The number of also_viewed_item neighbors is 9.
            Action 2: Finish[9]
            
            Definition of the graph: There are two types of nodes in the graph: item and brand. 
            Item nodes have features: title, description, price, img, category. Brand nodes have features: name. 
            Item nodes are linked to their brand nodes, also_viewed_item nodes, buy_after_viewing_item nodes, also_bought_item nodes, bought_together_item nodes. Brand nodes are linked to their item nodes.
            Question: What is the brand of item Dr. Suess 19163 Dr. Seuss Puzzle 3 Pack Bundle?
            Thought 1: The question is asking information of a item node (Dr. Suess 19163 Dr. Seuss Puzzle 3 Pack Bundle). We need to find the node in the graph and then check the node's brand neighbor from the graph.
            Action 1: Neighbor[Retrieve[Dr. Suess 19163 Dr. Seuss Puzzle 3 Pack Bundle], brand]
            Observation 1: The ID of this node is 0000191639. ['brand_5']
            Thought 2: The brand of item is ['brand_5']. The ID of this brand node is brand_5. We need to check the brand node's feature (name) from the graph.
            Action 2: Feature[brand_5, name]
            Observation 2: Dr. Seuss.
            Thought 3: The brand name of the item is Dr. Seuss.
            Action 3: Finish[Dr. Seuss]

            Definition of the graph: There are two types of nodes in the graph: item and brand. 
            Item nodes have features: title, description, price, img, category. Brand nodes have features: name. 
            Item nodes are linked to their brand nodes, also_viewed_item nodes, buy_after_viewing_item nodes, also_bought_item nodes, bought_together_item nodes. Brand nodes are linked to their item nodes.
            Question: What is the price of the item Purple Sequin Tiny Dancer Tutu Ballet Dance Fairy Princess Costume Accessory?
            Thought 1: The question is asking some basic information of a item node (Purple Sequin Tiny Dancer Tutu Ballet Dance Fairy Princess Costume Accessory). We need to find the node in the graph.
            Action 1: Retrieve[Purple Sequin Tiny Dancer Tutu Ballet Dance Fairy Princess Costume Accessory]
            Observation 1: The ID of this node is 0000037214.
            Thought 2: The question is asking the price of a item, we need to check the node feature (price) from the graph.
            Action 2: Feature[0000037214, price]
            Observation 2: 6.99
            Thought 3: The price of the item is 6.99.
            Action 3: Finish[6.99]
            
            Definition of the graph: There are two types of nodes in the graph: item and brand. 
            Item nodes have features: title, description, price, img, category. Brand nodes have features: name. 
            Item nodes are linked to their brand nodes, also_viewed_item nodes, buy_after_viewing_item nodes, also_bought_item nodes, bought_together_item nodes. Brand nodes are linked to their item nodes.
            Question: How many also_viewed_item does item The Sherlock Holmes Audio Collection have?
            Thought 1: The question is asking information of a item node (The Sherlock Holmes Audio Collection). We need to find the node in the graph.
            Action 1: Retrieve[The Sherlock Holmes Audio Collection]
            Observation 1: The ID of this node is 0001048236.
            Thought 2: The question is asking the number of also_viewed_item of an item, we need to calculate the node's also_viewed_item neighbor degree from the graph.
            Action 2: Degree[0001048236, also_viewed_item]
            Observation 2: 9
            Thought 3: The number of also_viewed_item neighbors is 9.
            Action 3: Finish[9]
            
            Definition of the graph: There are two types of nodes in the graph: item and brand. 
            Item nodes have features: title, description, price, img, category. Brand nodes have features: name. 
            Item nodes are linked to their brand nodes, also_viewed_item nodes, buy_after_viewing_item nodes, also_bought_item nodes, bought_together_item nodes. Brand nodes are linked to their item nodes.
            Question: What is the brand of item Dr. Suess 19163 Dr. Seuss Puzzle 3 Pack Bundle?
            Thought 1: The question is asking information of a item node (Dr. Suess 19163 Dr. Seuss Puzzle 3 Pack Bundle). We need to find the node in the graph.
            Action 1: Retrieve[Dr. Suess 19163 Dr. Seuss Puzzle 3 Pack Bundle]
            Observation 1: The ID of this node is 0000191639.
            Thought 2: The question is asking the brand of a item, we need to check the node's brand neighbor from the graph.
            Action 2: Neighbor[0000191639, brand]
            Observation 2: ['brand_5']
            Thought 3: The brand of item is ['brand_5']. The ID of this brand node is brand_5. We need to check the brand node's feature (name) from the graph.
            Action 3: Feature[brand_5, name]
            Observation 3: Dr. Seuss.
            Thought 4: The brand name of the item is Dr. Seuss.
            Action 4: Finish[Dr. Seuss]
            """,
    "biomedical": """Definition of the graph: There are eleven types of nodes in the graph: Anatomy, Biological Process, Cellular Component, Compound, Disease, Gene, Molecular Function, Pathway, Pharmacologic Class, Side Effect, Symptom.
        Each node has name feature.
        There are these types of edges: Anatomy-downregulates-Gene, Anatomy-expresses-Gene, Anatomy-upregulates-Gene, Compound-binds-Gene, Compound-causes-Side Effect, Compound-downregulates-Gene, Compound-palliates-Disease, Compound-resembles-Compound, Compound-treats-Disease, Compound-upregulates-Gene, Disease-associates-Gene, Disease-downregulates-Gene, Disease-localizes-Anatomy, Disease-presents-Symptom, Disease-resembles-Disease, Disease-upregulates-Gene, Gene-covaries-Gene, Gene-interacts-Gene, Gene-participates-Biological Process, Gene-participates-Cellular Component, Gene-participates-Molecular Function, Gene-participates-Pathway, Gene-regulates-Gene, Pharmacologic Class-includes-Compound.
        Question: What compounds can be used to treat Crohn's disease? Please answer the compound names rather than IDs.
        Thought 1: The question is related to a disease node (Crohn's disease). We need to find the node in the graph.
        Action 1: Retrieve[Crohn's disease]
        Observation 1: The ID of this node is DOID:8778.
        Thought 2: The question is asking the compounds which can be used to treat a disease, we need to check the node's 'Compound-treats-Disease' neighbor from the graph and get their names.
        Action 2: Feature[Neighbor[DOID:8778, Compound-treats-Disease], name]
        Observation 2: ['DB01014', 'DB00244', 'DB00795', 'DB00993', 'DB00635', 'DB01033'] Balsalazide, Balsalazide, Mesalazine, Sulfasalazine, Azathioprine, Prednisone, Mercaptopurine
        Thought 3: The name of compounds are Balsalazide, Mesalazine, Sulfasalazine, Azathioprine, Prednisone, Mercaptopurine.
        Action 3: Finish[Balsalazide, Mesalazine, Sulfasalazine, Azathioprine, Prednisone, Mercaptopurine]
        
        Definition of the graph: There are eleven types of nodes in the graph: Anatomy, Biological Process, Cellular Component, Compound, Disease, Gene, Molecular Function, Pathway, Pharmacologic Class, Side Effect, Symptom.
        Each node has name feature.
        There are these types of edges: Anatomy-downregulates-Gene, Anatomy-expresses-Gene, Anatomy-upregulates-Gene, Compound-binds-Gene, Compound-causes-Side Effect, Compound-downregulates-Gene, Compound-palliates-Disease, Compound-resembles-Compound, Compound-treats-Disease, Compound-upregulates-Gene, Disease-associates-Gene, Disease-downregulates-Gene, Disease-localizes-Anatomy, Disease-presents-Symptom, Disease-resembles-Disease, Disease-upregulates-Gene, Gene-covaries-Gene, Gene-interacts-Gene, Gene-participates-Biological Process, Gene-participates-Cellular Component, Gene-participates-Molecular Function, Gene-participates-Pathway, Gene-regulates-Gene, Pharmacologic Class-includes-Compound.
        Question: What is the inchikey of Caffeine?
        Thought 1: The question is related to a compound node (Caffeine). We need to find the node in the graph and check the node's 'inchikey' feature from the graph.
        Action 1: Feature[Retrieve[Caffeine], inchikey]
        Observation 1: The ID of this node is DB00201. InChIKey=RYYVLZVUVIJVGH-UHFFFAOYSA-N
        Thought 2:  The inchikey of the node is InChIKey=RYYVLZVUVIJVGH-UHFFFAOYSA-N.
        Action 2: Finish[InChIKey=RYYVLZVUVIJVGH-UHFFFAOYSA-N]
        
        Definition of the graph: There are eleven types of nodes in the graph: Anatomy, Biological Process, Cellular Component, Compound, Disease, Gene, Molecular Function, Pathway, Pharmacologic Class, Side Effect, Symptom.
        Each node has name feature.
        There are these types of edges: Anatomy-downregulates-Gene, Anatomy-expresses-Gene, Anatomy-upregulates-Gene, Compound-binds-Gene, Compound-causes-Side Effect, Compound-downregulates-Gene, Compound-palliates-Disease, Compound-resembles-Compound, Compound-treats-Disease, Compound-upregulates-Gene, Disease-associates-Gene, Disease-downregulates-Gene, Disease-localizes-Anatomy, Disease-presents-Symptom, Disease-resembles-Disease, Disease-upregulates-Gene, Gene-covaries-Gene, Gene-interacts-Gene, Gene-participates-Biological Process, Gene-participates-Cellular Component, Gene-participates-Molecular Function, Gene-participates-Pathway, Gene-regulates-Gene, Pharmacologic Class-includes-Compound.
        Question: How many side effects does Caffeine have?
        Thought 1: The question is related to a compound node (Caffeine). We need to find the node in the graph and calculate the number of the node's 'Compound-causes-Side Effect' neighbors from the graph.
        Action 1: Degree[Retrieve[Caffeine], 'Compound-causes-Side Effect']
        Observation 1: The ID of this node is DB00201. 58
        Thought 2: TThe number of 'Compound-causes-Side Effect' neighbors are 58.
        Action 2: Finish[58]

        Definition of the graph: There are eleven types of nodes in the graph: Anatomy, Biological Process, Cellular Component, Compound, Disease, Gene, Molecular Function, Pathway, Pharmacologic Class, Side Effect, Symptom.
        Each node has name feature.
        There are these types of edges: Anatomy-downregulates-Gene, Anatomy-expresses-Gene, Anatomy-upregulates-Gene, Compound-binds-Gene, Compound-causes-Side Effect, Compound-downregulates-Gene, Compound-palliates-Disease, Compound-resembles-Compound, Compound-treats-Disease, Compound-upregulates-Gene, Disease-associates-Gene, Disease-downregulates-Gene, Disease-localizes-Anatomy, Disease-presents-Symptom, Disease-resembles-Disease, Disease-upregulates-Gene, Gene-covaries-Gene, Gene-interacts-Gene, Gene-participates-Biological Process, Gene-participates-Cellular Component, Gene-participates-Molecular Function, Gene-participates-Pathway, Gene-regulates-Gene, Pharmacologic Class-includes-Compound.
        Question: What compounds can be used to treat Crohn's disease? Please answer the compound names rather than IDs.
        Thought 1: The question is related to a disease node (Crohn's disease). We need to find the node in the graph.
        Action 1: Retrieve[Crohn's disease]
        Observation 1: The ID of this node is DOID:8778.
        Thought 2: The question is asking the compounds which can be used to treat a disease, we need to check the node's 'Compound-treats-Disease' neighbor from the graph.
        Action 2: Neighbor[DOID:8778, Compound-treats-Disease]
        Observation 2: ['DB01014', 'DB00244', 'DB00795', 'DB00993', 'DB00635', 'DB01033']
        Thought 3: The IDs of the compounds are 'DB01014', 'DB00244', 'DB00795', 'DB00993', 'DB00635', 'DB01033'. We need to get their names.
        Action 3: Feature[DB01014, name], Feature[DB00244, name], Feature[DB00795, name], Feature[DB00993, name], Feature[DB00635, name], Feature[DB01033, name]
        Observation 3: Balsalazide, Balsalazide, Mesalazine, Sulfasalazine, Azathioprine, Prednisone, Mercaptopurine
        Thought 4: The name of compounds are Balsalazide, Mesalazine, Sulfasalazine, Azathioprine, Prednisone, Mercaptopurine.
        Action 4: Finish[Balsalazide, Mesalazine, Sulfasalazine, Azathioprine, Prednisone, Mercaptopurine]
        
        Definition of the graph: There are eleven types of nodes in the graph: Anatomy, Biological Process, Cellular Component, Compound, Disease, Gene, Molecular Function, Pathway, Pharmacologic Class, Side Effect, Symptom.
        Each node has name feature.
        There are these types of edges: Anatomy-downregulates-Gene, Anatomy-expresses-Gene, Anatomy-upregulates-Gene, Compound-binds-Gene, Compound-causes-Side Effect, Compound-downregulates-Gene, Compound-palliates-Disease, Compound-resembles-Compound, Compound-treats-Disease, Compound-upregulates-Gene, Disease-associates-Gene, Disease-downregulates-Gene, Disease-localizes-Anatomy, Disease-presents-Symptom, Disease-resembles-Disease, Disease-upregulates-Gene, Gene-covaries-Gene, Gene-interacts-Gene, Gene-participates-Biological Process, Gene-participates-Cellular Component, Gene-participates-Molecular Function, Gene-participates-Pathway, Gene-regulates-Gene, Pharmacologic Class-includes-Compound.
        Question: What is the inchikey of Caffeine?
        Thought 1: The question is related to a compound node (Caffeine). We need to find the node in the graph.
        Action 1: Retrieve[Caffeine]
        Observation 1: The ID of this node is DB00201.
        Thought 2: The question is asking the inchikey feature of a node, we need to check the node's 'inchikey' feature from the graph.
        Action 2: Feature[DB00201, inchikey]
        Observation 2: InChIKey=RYYVLZVUVIJVGH-UHFFFAOYSA-N
        Thought 3: The inchikey of the node is InChIKey=RYYVLZVUVIJVGH-UHFFFAOYSA-N.
        Action 3: Finish[InChIKey=RYYVLZVUVIJVGH-UHFFFAOYSA-N]
        
        Definition of the graph: There are eleven types of nodes in the graph: Anatomy, Biological Process, Cellular Component, Compound, Disease, Gene, Molecular Function, Pathway, Pharmacologic Class, Side Effect, Symptom.
        Each node has name feature.
        There are these types of edges: Anatomy-downregulates-Gene, Anatomy-expresses-Gene, Anatomy-upregulates-Gene, Compound-binds-Gene, Compound-causes-Side Effect, Compound-downregulates-Gene, Compound-palliates-Disease, Compound-resembles-Compound, Compound-treats-Disease, Compound-upregulates-Gene, Disease-associates-Gene, Disease-downregulates-Gene, Disease-localizes-Anatomy, Disease-presents-Symptom, Disease-resembles-Disease, Disease-upregulates-Gene, Gene-covaries-Gene, Gene-interacts-Gene, Gene-participates-Biological Process, Gene-participates-Cellular Component, Gene-participates-Molecular Function, Gene-participates-Pathway, Gene-regulates-Gene, Pharmacologic Class-includes-Compound.
        Question: How many side effects does Caffeine have?
        Thought 1: The question is related to a compound node (Caffeine). We need to find the node in the graph.
        Action 1: Retrieve[Caffeine]
        Observation 1: The ID of this node is DB00201.
        Thought 2: The question is asking the number of side effects a compound has, we need to calculate the number of the node's 'Compound-causes-Side Effect' neighbors from the graph.
        Action 2: Degree[DB00201, 'Compound-causes-Side Effect']
        Observation 2: 58
        Thought 3: The number of 'Compound-causes-Side Effect' neighbors are 58.
        Action 3: Finish[58]
            """,
    "legal": """Definition of the graph: There are four types of nodes in the graph: opinion, opinion_cluster, docket, and court.
        Opinion nodes have features: plain_text. Opinion_cluster nodes have features: syllabus, judges, case_name, attorneys. Docket nodes have features: pacer_case_id, case_name. Court nodes have features: full_name, start_date, end_date, citation_string.
        Opinion nodes are linked to their reference nodes and cited_by nodes, as well as their opinion_cluster nodes. Opinion_cluster nodes are linked to opinion nodes and docket nodes. Docket nodes are linked to opinion_cluster nodes and court nodes. Court nodes are linked to docket nodes.
        Question: Who are the judges that involved in the opinion_cluster: <p>Constitutional law — Title of act — Mortgage—Deed—Liability of grantee — Act of June IB, 1878, P. L. BOB.</p> <p>1. The Act of June 12, 1878, P. L. 205, entitled “An Act to define the liability of grantees of real estate for the incumbrances thereon,” is sufficiently comprehensive in its title to cover both the first and second sections of the act, and does not violate art. Ill, sec. 3 of the constitution of Pennsylvania.</p> <p>2. Where an owner of land creates two mortgages thereon and then conveys the property subject to the mortgages, the grantee covenanting in the deed itself to assume and pay the mortgages as a part of the consideration, and thereafter the grantee conveys the land to another person without expressly assuming any continuing liability to pay the mortgages, such grantee is not liable to the mortgag^ for payment of the mortgages. £V</p>?
        Thought 1: The question is related to a opinion_cluster (<p>Constitutional law — Title of act — Mortgage—Deed—Liability of grantee — Act of June IB, 1878, P. L. BOB.</p> <p>1. The Act of June 12, 1878, P. L. 205, entitled “An Act to define the liability of grantees of real estate for the incumbrances thereon,” is sufficiently comprehensive in its title to cover both the first and second sections of the act, and does not violate art. Ill, sec. 3 of the constitution of Pennsylvania.</p> <p>2. Where an owner of land creates two mortgages thereon and then conveys the property subject to the mortgages, the grantee covenanting in the deed itself to assume and pay the mortgages as a part of the consideration, and thereafter the grantee conveys the land to another person without expressly assuming any continuing liability to pay the mortgages, such grantee is not liable to the mortgag^ for payment of the mortgages. £V</p>). We need to find the node in the graph.
        Action 1: Retrieve[<p>Constitutional law — Title of act — Mortgage—Deed—Liability of grantee — Act of June IB, 1878, P. L. BOB.</p> <p>1. The Act of June 12, 1878, P. L. 205, entitled “An Act to define the liability of grantees of real estate for the incumbrances thereon,” is sufficiently comprehensive in its title to cover both the first and second sections of the act, and does not violate art. Ill, sec. 3 of the constitution of Pennsylvania.</p> <p>2. Where an owner of land creates two mortgages thereon and then conveys the property subject to the mortgages, the grantee covenanting in the deed itself to assume and pay the mortgages as a part of the consideration, and thereafter the grantee conveys the land to another person without expressly assuming any continuing liability to pay the mortgages, such grantee is not liable to the mortgag^ for payment of the mortgages. £V</p>]
        Observation 1: The ID of this node is opc-6381448.
        Thought 2: The question is asking the judges of this opinion_cluster node, we need to check the node's 'judges' feature from the graph.
        Action 2: Feature[opc-6381448, judges]
        Observation 2: Brown, Elkin, Fell, Mestrezat, Moschzisker, Potter, Stewart
        Thought 3: The judges of this opinion cluster is Brown, Elkin, Fell, Mestrezat, Moschzisker, Potter, Stewart
        Action 3: Finish[Brown, Elkin, Fell, Mestrezat, Moschzisker, Potter, Stewart]

        Definition of the graph: There are four types of nodes in the graph: opinion, opinion_cluster, docket, and court.
        Opinion nodes have features: plain_text. Opinion_cluster nodes have features: syllabus, judges, case_name, attorneys. Docket nodes have features: pacer_case_id, case_name. Court nodes have features: full_name, start_date, end_date, citation_string.
        Opinion nodes are linked to their reference nodes and cited_by nodes, as well as their opinion_cluster nodes. Opinion_cluster nodes are linked to opinion nodes and docket nodes. Docket nodes are linked to opinion_cluster nodes and court nodes. Court nodes are linked to docket nodes.
        Question: How many opinions are contained in this opinion_cluster: <p>A general statement that the decision of a state court is against the constitutional rights of the objecting party, or against the Fourteenth Amendment, or that it is without due process of law,-particularly when these objections appear only in specifications of error, so called, will not raise a Federal question, even where the judgment is a final one within Rev. Stat. § 709.</p> <p>In these cases there was no final judgment, such as is provided for in Rev. Stat. § 709, and there does not appear to have arisen any Federal question whatever.</p>'},
        Thought 1: The question is related to a opinion_cluster (<p>A general statement that the decision of a state court is against the constitutional rights of the objecting party, or against the Fourteenth Amendment, or that it is without due process of law,-particularly when these objections appear only in specifications of error, so called, will not raise a Federal question, even where the judgment is a final one within Rev. Stat. § 709.</p> <p>In these cases there was no final judgment, such as is provided for in Rev. Stat. § 709, and there does not appear to have arisen any Federal question whatever.</p>). We need to find the node in the graph.
        Action 1: Retrieve[<p>A general statement that the decision of a state court is against the constitutional rights of the objecting party, or against the Fourteenth Amendment, or that it is without due process of law,-particularly when these objections appear only in specifications of error, so called, will not raise a Federal question, even where the judgment is a final one within Rev. Stat. § 709.</p> <p>In these cases there was no final judgment, such as is provided for in Rev. Stat. § 709, and there does not appear to have arisen any Federal question whatever.</p>]
        Observation 1: The ID of this node is opc-94592.
        Thought 2: The question is asking the number of the opinion neighbors of this node, we need to calculate the node's "opinion" neighbor degree from the graph.
        Action 2: Degree[opc-94592, opinion]
        Observation 2: 1
        Thought 3: The number of the opinion neighbors is 1
        Action 3: Finish[1]

        Definition of the graph: There are four types of nodes in the graph: opinion, opinion_cluster, docket, and court.
        Opinion nodes have features: plain_text. Opinion_cluster nodes have features: syllabus, judges, case_name, attorneys. Docket nodes have features: pacer_case_id, case_name. Court nodes have features: full_name, start_date, end_date, citation_string.
        Opinion nodes are linked to their reference nodes and cited_by nodes, as well as their opinion_cluster nodes. Opinion_cluster nodes are linked to opinion nodes and docket nodes. Docket nodes are linked to opinion_cluster nodes and court nodes. Court nodes are linked to docket nodes.
        Question: What is the docket pacer case ID for this opinion cluster: In May, 1873, a brewer pays to the collector $100 as special tax, for which a special-tax stamp is issued to Mm. Subsequently he shoivs to. the satisfaction of the Commissions)\' of Internal Revenue that he was not a brewer of more than 500 barrels, and henee that he is entitled to have $50 refunded. The Commissioner issues his certificate for a special-tax stamp of the value of%50 returned. The Treasury refuses payment of the certificate. The breioe)\' brings his action to recover the amount thereof.</p> <p>I. The Commissioner of Internal Revenue being authorized by one section of the statutes (Rev. Stat., § 3220) to refund taxes “unjustly assessed or excessive in, amount," and by another, (§ 3426,) the value of special-tax stamps ‘‘where the rates of duties represented thereby have been paid in error or remitted," it is immaterial whether his certificate be in form under the one section or the other so long as it represents a case where the Commissioner is authorized to refund and the amoirnt which the party is entitled to have refunded.</p> <p>II. When a controversy under the revenue laws is by law required to be determined by other officers and other tribunals, this court is without jurisdiction; but where the proper officer has determined a question, and awarded an allowance, and filed a certificate, and exhausted his jurisdiction, and the Treasury has refused to carry out the award by mating payment, this court has jurisdiction of an action brought to recover the. award..
        Thought 1: The question is related to a opinion cluster (In May, 1873, a brewer pays to the collector $100 as special tax, for which a special-tax stamp is issued to Mm. Subsequently he shoivs to. the satisfaction of the Commissions)\' of Internal Revenue that he was not a brewer of more than 500 barrels, and henee that he is entitled to have $50 refunded. The Commissioner issues his certificate for a special-tax stamp of the value of%50 returned. The Treasury refuses payment of the certificate. The breioe)\' brings his action to recover the amount thereof.</p> <p>I. The Commissioner of Internal Revenue being authorized by one section of the statutes (Rev. Stat., § 3220) to refund taxes “unjustly assessed or excessive in, amount," and by another, (§ 3426,) the value of special-tax stamps ‘‘where the rates of duties represented thereby have been paid in error or remitted," it is immaterial whether his certificate be in form under the one section or the other so long as it represents a case where the Commissioner is authorized to refund and the amoirnt which the party is entitled to have refunded.</p> <p>II. When a controversy under the revenue laws is by law required to be determined by other officers and other tribunals, this court is without jurisdiction; but where the proper officer has determined a question, and awarded an allowance, and filed a certificate, and exhausted his jurisdiction, and the Treasury has refused to carry out the award by mating payment, this court has jurisdiction of an action brought to recover the. award.). We need to find the node in the graph.
        Action 1: Retrieve[In May, 1873, a brewer pays to the collector $100 as special tax, for which a special-tax stamp is issued to Mm. Subsequently he shoivs to. the satisfaction of the Commissions)\' of Internal Revenue that he was not a brewer of more than 500 barrels, and henee that he is entitled to have $50 refunded. The Commissioner issues his certificate for a special-tax stamp of the value of%50 returned. The Treasury refuses payment of the certificate. The breioe)\' brings his action to recover the amount thereof.</p> <p>I. The Commissioner of Internal Revenue being authorized by one section of the statutes (Rev. Stat., § 3220) to refund taxes “unjustly assessed or excessive in, amount," and by another, (§ 3426,) the value of special-tax stamps ‘‘where the rates of duties represented thereby have been paid in error or remitted," it is immaterial whether his certificate be in form under the one section or the other so long as it represents a case where the Commissioner is authorized to refund and the amoirnt which the party is entitled to have refunded.</p> <p>II. When a controversy under the revenue laws is by law required to be determined by other officers and other tribunals, this court is without jurisdiction; but where the proper officer has determined a question, and awarded an allowance, and filed a certificate, and exhausted his jurisdiction, and the Treasury has refused to carry out the award by mating payment, this court has jurisdiction of an action brought to recover the. award.]
        Observation 1: The ID of this node is opc-8599951.
        Thought 2: The question is asking the judges of this opinion cluster node, we need to check the node's 'judges' feature from the graph.
        Action 2: Feature[opc-8599951]
        Observation 2: {'judges': 'Eichaedson', 'case_name': 'Kaufman v. United States', 'attorneys': 'Mr. P. P. Pye for the claimant., Mr. Assistant Attorney-General Simons for the defendants.', 'syllabus': '<p>In May, 1873, a brewer pays to the collector $100 as special tax, for which a special-tax stamp is issued to Mm. Subsequently he shoivs to. the satisfaction of the Commissions)\' of Internal Revenue that he was not a brewer of more than 500 barrels, and henee that he is entitled to have $50 refunded. The Commissioner issues his certificate for a special-tax stamp of the value of%50 returned. The Treasury refuses payment of the certificate. The breioe)\' brings his action to recover the amount thereof.</p> <p>I. The Commissioner of Internal Revenue being authorized by one section of the statutes (Rev. Stat., § 3220) to refund taxes “unjustly assessed or excessive in, amount," and by another, (§ 3426,) the value of special-tax stamps ‘‘where the rates of duties represented thereby have been paid in error or remitted," it is immaterial whether his certificate be in form under the one section or the other so long as it represents a case where the Commissioner is authorized to refund and the amoirnt which the party is entitled to have refunded.</p> <p>II. When a controversy under the revenue laws is by law required to be determined by other officers and other tribunals, this court is without jurisdiction; but where the proper officer has determined a question, and awarded an allowance, and filed a certificate, and exhausted his jurisdiction, and the Treasury has refused to carry out the award by mating payment, this court has jurisdiction of an action brought to recover the. award.</p>'}
        Thought 3: The judges of this opinion cluster is Eichaedson
        Action 3: Finish[Eichaedson]

        Definition of the graph: There are four types of nodes in the graph: opinion, opinion_cluster, docket, and court.
        Opinion nodes have features: plain_text. Opinion_cluster nodes have features: syllabus, judges, case_name, attorneys. Docket nodes have features: pacer_case_id, case_name. Court nodes have features: full_name, start_date, end_date, citation_string.
        Opinion nodes are linked to their reference nodes and cited_by nodes, as well as their opinion_cluster nodes. Opinion_cluster nodes are linked to opinion nodes and docket nodes. Docket nodes are linked to opinion_cluster nodes and court nodes. Court nodes are linked to docket nodes.
        Question: Who are the judges that involved in the opinion_cluster: <p>Constitutional law — Title of act — Mortgage—Deed—Liability of grantee — Act of June IB, 1878, P. L. BOB.</p> <p>1. The Act of June 12, 1878, P. L. 205, entitled “An Act to define the liability of grantees of real estate for the incumbrances thereon,” is sufficiently comprehensive in its title to cover both the first and second sections of the act, and does not violate art. Ill, sec. 3 of the constitution of Pennsylvania.</p> <p>2. Where an owner of land creates two mortgages thereon and then conveys the property subject to the mortgages, the grantee covenanting in the deed itself to assume and pay the mortgages as a part of the consideration, and thereafter the grantee conveys the land to another person without expressly assuming any continuing liability to pay the mortgages, such grantee is not liable to the mortgag^ for payment of the mortgages. £V</p>?
        Thought 1: The question is related to a opinion_cluster (<p>Constitutional law — Title of act — Mortgage—Deed—Liability of grantee — Act of June IB, 1878, P. L. BOB.</p> <p>1. The Act of June 12, 1878, P. L. 205, entitled “An Act to define the liability of grantees of real estate for the incumbrances thereon,” is sufficiently comprehensive in its title to cover both the first and second sections of the act, and does not violate art. Ill, sec. 3 of the constitution of Pennsylvania.</p> <p>2. Where an owner of land creates two mortgages thereon and then conveys the property subject to the mortgages, the grantee covenanting in the deed itself to assume and pay the mortgages as a part of the consideration, and thereafter the grantee conveys the land to another person without expressly assuming any continuing liability to pay the mortgages, such grantee is not liable to the mortgag^ for payment of the mortgages. £V</p>). We need to find the node in the graph. The question is asking the judges of this opinion_cluster node, we need to check the node's 'judges' feature from the graph.
        Action 1: Feature[Retrieve[<p>Constitutional law — Title of act — Mortgage—Deed—Liability of grantee — Act of June IB, 1878, P. L. BOB.</p> <p>1. The Act of June 12, 1878, P. L. 205, entitled “An Act to define the liability of grantees of real estate for the incumbrances thereon,” is sufficiently comprehensive in its title to cover both the first and second sections of the act, and does not violate art. Ill, sec. 3 of the constitution of Pennsylvania.</p> <p>2. Where an owner of land creates two mortgages thereon and then conveys the property subject to the mortgages, the grantee covenanting in the deed itself to assume and pay the mortgages as a part of the consideration, and thereafter the grantee conveys the land to another person without expressly assuming any continuing liability to pay the mortgages, such grantee is not liable to the mortgag^ for payment of the mortgages. £V</p>], judges]
        Observation 1: The ID of this node is opc-6381448.  Brown, Elkin, Fell, Mestrezat, Moschzisker, Potter, Stewart
        Thought 2: The judges of this opinion cluster is Brown, Elkin, Fell, Mestrezat, Moschzisker, Potter, Stewart
        Action 2: Finish[Brown, Elkin, Fell, Mestrezat, Moschzisker, Potter, Stewart]

        Definition of the graph: There are four types of nodes in the graph: opinion, opinion_cluster, docket, and court.
        Opinion nodes have features: plain_text. Opinion_cluster nodes have features: syllabus, judges, case_name, attorneys. Docket nodes have features: pacer_case_id, case_name. Court nodes have features: full_name, start_date, end_date, citation_string.
        Opinion nodes are linked to their reference nodes and cited_by nodes, as well as their opinion_cluster nodes. Opinion_cluster nodes are linked to opinion nodes and docket nodes. Docket nodes are linked to opinion_cluster nodes and court nodes. Court nodes are linked to docket nodes.
        Question: How many opinions are contained in this opinion_cluster: <p>A general statement that the decision of a state court is against the constitutional rights of the objecting party, or against the Fourteenth Amendment, or that it is without due process of law,-particularly when these objections appear only in specifications of error, so called, will not raise a Federal question, even where the judgment is a final one within Rev. Stat. § 709.</p> <p>In these cases there was no final judgment, such as is provided for in Rev. Stat. § 709, and there does not appear to have arisen any Federal question whatever.</p>'},
        Thought 1: The question is related to a opinion_cluster (<p>A general statement that the decision of a state court is against the constitutional rights of the objecting party, or against the Fourteenth Amendment, or that it is without due process of law,-particularly when these objections appear only in specifications of error, so called, will not raise a Federal question, even where the judgment is a final one within Rev. Stat. § 709.</p> <p>In these cases there was no final judgment, such as is provided for in Rev. Stat. § 709, and there does not appear to have arisen any Federal question whatever.</p>). We need to find the node in the graph. The question is asking the number of the opinion neighbors of this node, we need to calculate the node's "opinion" neighbor degree from the graph.
        Action 1: Degree[Retrieve[<p>A general statement that the decision of a state court is against the constitutional rights of the objecting party, or against the Fourteenth Amendment, or that it is without due process of law,-particularly when these objections appear only in specifications of error, so called, will not raise a Federal question, even where the judgment is a final one within Rev. Stat. § 709.</p> <p>In these cases there was no final judgment, such as is provided for in Rev. Stat. § 709, and there does not appear to have arisen any Federal question whatever.</p>], opinion]
        Observation 1: The ID of this node is opc-94592. 1
        Thought 2: The number of the opinion neighbors is 1
        Action 2: Finish[1]

        Definition of the graph: There are four types of nodes in the graph: opinion, opinion_cluster, docket, and court.
        Opinion nodes have features: plain_text. Opinion_cluster nodes have features: syllabus, judges, case_name, attorneys. Docket nodes have features: pacer_case_id, case_name. Court nodes have features: full_name, start_date, end_date, citation_string.
        Opinion nodes are linked to their reference nodes and cited_by nodes, as well as their opinion_cluster nodes. Opinion_cluster nodes are linked to opinion nodes and docket nodes. Docket nodes are linked to opinion_cluster nodes and court nodes. Court nodes are linked to docket nodes.
        Question: What is the docket pacer case ID for this opinion cluster: In May, 1873, a brewer pays to the collector $100 as special tax, for which a special-tax stamp is issued to Mm. Subsequently he shoivs to. the satisfaction of the Commissions)\' of Internal Revenue that he was not a brewer of more than 500 barrels, and henee that he is entitled to have $50 refunded. The Commissioner issues his certificate for a special-tax stamp of the value of%50 returned. The Treasury refuses payment of the certificate. The breioe)\' brings his action to recover the amount thereof.</p> <p>I. The Commissioner of Internal Revenue being authorized by one section of the statutes (Rev. Stat., § 3220) to refund taxes “unjustly assessed or excessive in, amount," and by another, (§ 3426,) the value of special-tax stamps ‘‘where the rates of duties represented thereby have been paid in error or remitted," it is immaterial whether his certificate be in form under the one section or the other so long as it represents a case where the Commissioner is authorized to refund and the amoirnt which the party is entitled to have refunded.</p> <p>II. When a controversy under the revenue laws is by law required to be determined by other officers and other tribunals, this court is without jurisdiction; but where the proper officer has determined a question, and awarded an allowance, and filed a certificate, and exhausted his jurisdiction, and the Treasury has refused to carry out the award by mating payment, this court has jurisdiction of an action brought to recover the. award..
        Thought 1: The question is related to a opinion cluster (In May, 1873, a brewer pays to the collector $100 as special tax, for which a special-tax stamp is issued to Mm. Subsequently he shoivs to. the satisfaction of the Commissions)\' of Internal Revenue that he was not a brewer of more than 500 barrels, and henee that he is entitled to have $50 refunded. The Commissioner issues his certificate for a special-tax stamp of the value of%50 returned. The Treasury refuses payment of the certificate. The breioe)\' brings his action to recover the amount thereof.</p> <p>I. The Commissioner of Internal Revenue being authorized by one section of the statutes (Rev. Stat., § 3220) to refund taxes “unjustly assessed or excessive in, amount," and by another, (§ 3426,) the value of special-tax stamps ‘‘where the rates of duties represented thereby have been paid in error or remitted," it is immaterial whether his certificate be in form under the one section or the other so long as it represents a case where the Commissioner is authorized to refund and the amoirnt which the party is entitled to have refunded.</p> <p>II. When a controversy under the revenue laws is by law required to be determined by other officers and other tribunals, this court is without jurisdiction; but where the proper officer has determined a question, and awarded an allowance, and filed a certificate, and exhausted his jurisdiction, and the Treasury has refused to carry out the award by mating payment, this court has jurisdiction of an action brought to recover the. award.). We need to find the node in the graph. The question is asking the judges of this opinion cluster node, we need to check the node's 'judges' feature from the graph.
        Action 1: Feature[Retrieve[In May, 1873, a brewer pays to the collector $100 as special tax, for which a special-tax stamp is issued to Mm. Subsequently he shoivs to. the satisfaction of the Commissions)\' of Internal Revenue that he was not a brewer of more than 500 barrels, and henee that he is entitled to have $50 refunded. The Commissioner issues his certificate for a special-tax stamp of the value of%50 returned. The Treasury refuses payment of the certificate. The breioe)\' brings his action to recover the amount thereof.</p> <p>I. The Commissioner of Internal Revenue being authorized by one section of the statutes (Rev. Stat., § 3220) to refund taxes “unjustly assessed or excessive in, amount," and by another, (§ 3426,) the value of special-tax stamps ‘‘where the rates of duties represented thereby have been paid in error or remitted," it is immaterial whether his certificate be in form under the one section or the other so long as it represents a case where the Commissioner is authorized to refund and the amoirnt which the party is entitled to have refunded.</p> <p>II. When a controversy under the revenue laws is by law required to be determined by other officers and other tribunals, this court is without jurisdiction; but where the proper officer has determined a question, and awarded an allowance, and filed a certificate, and exhausted his jurisdiction, and the Treasury has refused to carry out the award by mating payment, this court has jurisdiction of an action brought to recover the. award.]]
        Observation 1: The ID of this node is opc-8599951. {'judges': 'Eichaedson', 'case_name': 'Kaufman v. United States', 'attorneys': 'Mr. P. P. Pye for the claimant., Mr. Assistant Attorney-General Simons for the defendants.', 'syllabus': '<p>In May, 1873, a brewer pays to the collector $100 as special tax, for which a special-tax stamp is issued to Mm. Subsequently he shoivs to. the satisfaction of the Commissions)\' of Internal Revenue that he was not a brewer of more than 500 barrels, and henee that he is entitled to have $50 refunded. The Commissioner issues his certificate for a special-tax stamp of the value of%50 returned. The Treasury refuses payment of the certificate. The breioe)\' brings his action to recover the amount thereof.</p> <p>I. The Commissioner of Internal Revenue being authorized by one section of the statutes (Rev. Stat., § 3220) to refund taxes “unjustly assessed or excessive in, amount," and by another, (§ 3426,) the value of special-tax stamps ‘‘where the rates of duties represented thereby have been paid in error or remitted," it is immaterial whether his certificate be in form under the one section or the other so long as it represents a case where the Commissioner is authorized to refund and the amoirnt which the party is entitled to have refunded.</p> <p>II. When a controversy under the revenue laws is by law required to be determined by other officers and other tribunals, this court is without jurisdiction; but where the proper officer has determined a question, and awarded an allowance, and filed a certificate, and exhausted his jurisdiction, and the Treasury has refused to carry out the award by mating payment, this court has jurisdiction of an action brought to recover the. award.</p>'}
        Thought 2: The judges of this opinion cluster is Eichaedson
        Action 2: Finish[Eichaedson]
        """,
    "goodreads": """Definition of the graph: There are four types of nodes in the graph: book, author, publisher, and series. 
        Book nodes have features: country_code, language_code, is_ebook, title, description, format, num_pages, publication_year, url, popular_shelves, and genres. Author nodes have features: name. Publisher nodes have features: name. Series nodes have features: title and description. 
        Book nodes are linked to their author nodes, publisher nodes, series nodes and similar_books nodes. Author nodes are linked to their book nodes. Publisher nodes are linked to their book nodes. Series nodes are linked to their book nodes.
        Question: Who is the author of the book 'The Great Gatsby'?
        Thought 1: The question is asking for the author of a specific book (The Great Gatsby). We need to find the book node in the graph.
        Action 1: Retrieve[The Great Gatsby]
        Observation 1: The ID of this book node is 21366268.
        Thought 2: The question is asking for the author of the book. We need to check the book node's author neighbor from the graph.
        Action 2: Neighbor[21366268, author]
        Observation 2: ['3190']
        Thought 3: The ID of the author is 3190. We need to check the name.
        Action 3: Feature[3190, name]
        Observation 3: F. Scott Fitzgerald
        Thought 4: The author of 'The Great Gatsby' is F. Scott Fitzgerald.
        Action 4: Finish[F. Scott Fitzgerald]
        
        Definition of the graph: There are four types of nodes in the graph: book, author, publisher, and series. 
        Book nodes have features: country_code, language_code, is_ebook, title, description, format, num_pages, publication_year, url, popular_shelves, and genres. Author nodes have features: name. Publisher nodes have features: name. Series nodes have features: title and description. 
        Book nodes are linked to their author nodes, publisher nodes, series nodes and similar_books nodes. Author nodes are linked to their book nodes. Publisher nodes are linked to their book nodes. Series nodes are linked to their book nodes.
        Question: What is the genre of the book '1984'?
        Thought 1: The question is asking for the genre of a specific book (1984). We need to find the book node in the graph.
        Action 1: Retrieve[1984]
        Observation 1: The ID of this book node is 3475269.
        Thought 2: The question is asking for the genre of the book. We need to check the book node's genre feature from the graph.
        Action 2: Feature[3475269, genres]
        Observation 2: currently-reading, classics, fiction
        Thought 3: The genres of the book '1984' are currently-reading, classics, fiction.
        Action 3: Finish[currently-reading, classics, fiction]
        
        Definition of the graph: There are four types of nodes in the graph: book, author, publisher, and series. 
        Book nodes have features: country_code, language_code, is_ebook, title, description, format, num_pages, publication_year, url, popular_shelves, and genres. Author nodes have features: name. Publisher nodes have features: name. Series nodes have features: title and description. 
        Book nodes are linked to their author nodes, publisher nodes, series nodes and similar_books nodes. Author nodes are linked to their book nodes. Publisher nodes are linked to their book nodes. Series nodes are linked to their book nodes.
        Question: How many series included the book 'Harry Potter and the Sorcerer's Stone'?
        Thought 1: We need to find the book node for 'Harry Potter and the Sorcerer's Stone' to determine its series.
        Action 1: Retrieve[Harry Potter and the Sorcerer's Stone]
        Observation 1: The ID of this book node is 27421523.
        Thought 2: Now that we have the book node, we need to calculate the node's series neighbor degree from the graph.
        Action 2: Degree[27421523, series]
        Observation 2: 1
        Thought 3: The number of series neighbors is 1.
        Action 3: Finish[1]

        Definition of the graph: There are four types of nodes in the graph: book, author, publisher, and series. 
        Book nodes have features: country_code, language_code, is_ebook, title, description, format, num_pages, publication_year, url, popular_shelves, and genres. Author nodes have features: name. Publisher nodes have features: name. Series nodes have features: title and description. 
        Book nodes are linked to their author nodes, publisher nodes, series nodes and similar_books nodes. Author nodes are linked to their book nodes. Publisher nodes are linked to their book nodes. Series nodes are linked to their book nodes.
        Question: Who is the author of the book 'The Great Gatsby'?
        Thought 1: The question is asking for the author of a specific book (The Great Gatsby). We need to find the book node in the graph. The question is asking for the author of the book. We need to check the book node's author neighbor from the graph.
        Action 1: Neighbor[Retrieve[The Great Gatsby], author]
        Observation 1: The ID of this book node is 21366268. ['3190']
        Thought 2: The ID of the author is 3190. We need to check the name.
        Action 2: Feature[3190, name]
        Observation 2: F. Scott Fitzgerald
        Thought 3: The author of 'The Great Gatsby' is F. Scott Fitzgerald.
        Action 3: Finish[F. Scott Fitzgerald]
        
        Definition of the graph: There are four types of nodes in the graph: book, author, publisher, and series. 
        Book nodes have features: country_code, language_code, is_ebook, title, description, format, num_pages, publication_year, url, popular_shelves, and genres. Author nodes have features: name. Publisher nodes have features: name. Series nodes have features: title and description. 
        Book nodes are linked to their author nodes, publisher nodes, series nodes and similar_books nodes. Author nodes are linked to their book nodes. Publisher nodes are linked to their book nodes. Series nodes are linked to their book nodes.
        Question: What is the genre of the book '1984'?
        Thought 1: The question is asking for the genre of a specific book (1984). We need to find the book node in the graph. The question is asking for the genre of the book. We need to check the book node's genre feature from the graph.
        Action 1: Feature[Retrieve[1984], genres]
        Observation 1: The ID of this book node is 3475269. currently-reading, classics, fiction
        Thought 2: The genres of the book '1984' are currently-reading, classics, fiction.
        Action 2: Finish[currently-reading, classics, fiction]
        
        Definition of the graph: There are four types of nodes in the graph: book, author, publisher, and series. 
        Book nodes have features: country_code, language_code, is_ebook, title, description, format, num_pages, publication_year, url, popular_shelves, and genres. Author nodes have features: name. Publisher nodes have features: name. Series nodes have features: title and description. 
        Book nodes are linked to their author nodes, publisher nodes, series nodes and similar_books nodes. Author nodes are linked to their book nodes. Publisher nodes are linked to their book nodes. Series nodes are linked to their book nodes.
        Question: How many series included the book 'Harry Potter and the Sorcerer's Stone'?
        Thought 1: We need to find the book node for 'Harry Potter and the Sorcerer's Stone' to determine its series and calculate the node's series neighbor degree from the graph.
        Action 1: Degree[Retrieve[Harry Potter and the Sorcerer's Stone], series]
        Observation 1: The ID of this book node is 27421523. 1
        Thought 2: The number of series neighbors is 1.
        Action 2: Finish[1]
        """,
    "dblp": """Definition of the graph: There are three types of nodes in the graph: paper, author and venue.
        Paper nodes have features: title, abstract, keywords, lang, and year. Author nodes have features: name and organization. Venue nodes have features: name.
        Paper nodes are linked to their author nodes, venue nodes, reference nodes (the papers this paper cite) and cited_by nodes (other papers which cite this paper). Author nodes are linked to their paper nodes. Venue nodes are linked to their paper nodes.
        Question: What organization is researcher Greg Daville from?
        Thought 1: The question is asking about the organization of a researcher named Greg Daville. We need to find the node corresponding to Greg Daville in the graph.
        Action 1: Retrieve[Greg Daville]
        Observation 1: The ID of this retrieval target node is 53f460a7dabfaee4dc83702a.
        Thought 2: The question is asking for the organization of a researcher, so we need to check the node's organization feature from the graph.
        Action 2: Feature[53f460a7dabfaee4dc83702a, organization]
        Observation 2: Hove East Sussex, United Kingdom 
        Thought 3: The organization of researcher Greg Daville is Hove East Sussex, United Kingdom.
        Action 3: Finish[Hove East Sussex, United Kingdom] 
    
        Definition of the graph: There are three types of nodes in the graph: paper, author and venue.
        Paper nodes have features: title, abstract, keywords, lang, and year. Author nodes have features: name and organization. Venue nodes have features: name.
        Paper nodes are linked to their author nodes, venue nodes, reference nodes (the papers this paper cite) and cited_by nodes (other papers which cite this paper). Author nodes are linked to their paper nodes. Venue nodes are linked to their paper nodes.
        Question: How many papers are written by author Nicholas Lydon?
        Thought 1: The question is asking for the number of written papers of a specific author (Nicholas Lydon). We need to find the author node in the graph.
        Action 1: Retrieve[Nicholas Lydon]
        Observation 1: The ID of this retrieval target node is 53f438c3dabfaedf43596117.
        Thought 2: The question is asking for the number of papers written by Nicholas Lydon. We need to calculate the "paper" neighbor degree of this node.
        Action 2: Degree[53f438c3dabfaedf43596117, paper]
        Observation 2: 2
        Thought 3: The number of the paper neighbors is 2
        Action 3: Finish[2]        

        Definition of the graph: There are three types of nodes in the graph: paper, author and venue.
        Paper nodes have features: title, abstract, keywords, lang, and year. Author nodes have features: name and organization. Venue nodes have features: name.
        Paper nodes are linked to their author nodes, venue nodes, reference nodes (the papers this paper cite) and cited_by nodes (other papers which cite this paper). Author nodes are linked to their paper nodes. Venue nodes are linked to their paper nodes.
        Question: What was the publish venue of the paper Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars?
        Thought 1: The question is asking information of a node (Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars). We need to find the node in the graph.
        Action 1: Retrieve[Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars]
        Observation 1: The ID of this node is 2090642949.
        Thought 2: The question is asking the published venue of a paper, we need to check the node's venue neighbor from the graph.
        Action 2: Neighbor[2090642949, venue]
        Observation 2: ['1980519', '1053242']
        Thought 3: The ID of the published venue are 1980519 and 1053242. We need to get their names.
        Action 3: Feature[1980519, name], Feature[1053242, name]
        Observation 3: the astrophysical journal, the atmosphere journal
        Thought 4: The name of the published venues are the astrophysical journal and the atmosphere journal
        Action 4: Finish[the astrophysical journal, the atmosphere journal]

        Definition of the graph: There are three types of nodes in the graph: paper, author and venue.
        Paper nodes have features: title, abstract, keywords, lang, and year. Author nodes have features: name and organization. Venue nodes have features: name.
        Paper nodes are linked to their author nodes, venue nodes, reference nodes (the papers this paper cite) and cited_by nodes (other papers which cite this paper). Author nodes are linked to their paper nodes. Venue nodes are linked to their paper nodes.
        Question: What organization is researcher Greg Daville from?
        Thought 1: The question is asking about the organization of a researcher named Greg Daville. We need to find the node corresponding to Greg Daville in the graph. The question is asking for the organization of a researcher, so we need to check the node's organization feature from the graph.
        Action 1: Feature[Retrieve[Greg Daville], organization]
        Observation 1: The ID of this retrieval target node is 53f460a7dabfaee4dc83702a. Hove East Sussex, United Kingdom 
        Thought 2: The organization of researcher Greg Daville is Hove East Sussex, United Kingdom.
        Action 2: Finish[Hove East Sussex, United Kingdom]
    
        Definition of the graph: There are three types of nodes in the graph: paper, author and venue.
        Paper nodes have features: title, abstract, keywords, lang, and year. Author nodes have features: name and organization. Venue nodes have features: name.
        Paper nodes are linked to their author nodes, venue nodes, reference nodes (the papers this paper cite) and cited_by nodes (other papers which cite this paper). Author nodes are linked to their paper nodes. Venue nodes are linked to their paper nodes.
        Question: How many papers are written by author Nicholas Lydon?
        Thought 1: The question is asking for the number of written papers of a specific author (Nicholas Lydon). We need to find the author node in the graph. The question is asking for the number of papers written by Nicholas Lydon. We need to calculate the "paper" neighbor degree of this node.
        Action 1: Degree[Retrieve[Nicholas Lydon], paper]
        Observation 1: The ID of this retrieval target node is 53f438c3dabfaedf43596117. 2
        Thought 2: The number of the paper neighbors is 2
        Action 2: Finish[2]

        Definition of the graph: There are three types of nodes in the graph: paper, author and venue.
        Paper nodes have features: title, abstract, keywords, lang, and year. Author nodes have features: name and organization. Venue nodes have features: name.
        Paper nodes are linked to their author nodes, venue nodes, reference nodes (the papers this paper cite) and cited_by nodes (other papers which cite this paper). Author nodes are linked to their paper nodes. Venue nodes are linked to their paper nodes.
        Question: What was the publish venue of the paper Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars?
        Thought 1: The question is asking information of a node (Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars). We need to find the node in the graph.
        Action 1: Retrieve[Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars]
        Observation 1: The ID of this node is 2090642949.
        Thought 2: The question is asking the published venue of a paper, we need to check the node's venue neighbor from the graph. The ID of the published venue are 1980519 and 1053242. We need to get their names.
        Action 2: Feature[Neighbor[2090642949, venue], name]
        Observation 2: ['1980519', '1053242'] the astrophysical journal, the atmosphere journal
        Thought 3: The name of the published venues are the astrophysical journal and the atmosphere journal
        Action 3: Finish[the astrophysical journal, the atmosphere journal]
        """,
}

PLAN_SHORT_EXAMPLES = {
    "maple": """Definition of the graph: There are three types of nodes in the graph: paper, author and venue.
        Paper nodes have features: title, abstract, year and label. Author nodes have features: name. Venue nodes have features: name.
        Paper nodes are linked to their author nodes, venue nodes, reference paper nodes and cited_by paper nodes. Author nodes are linked to their paper nodes. Venue nodes are linked to their paper nodes.
        Question: When was the paper Strongly Interacting Higgs Sector in the Minimal Standard Model published?
        Plan 1: The question is asking the published time of a node (Strongly Interacting Higgs Sector in the Minimal Standard Model). Thus, we need to locate the relevant paper and extract the published year from the graph.
        Thought 1: We need to find the node in the graph and check the node feature (year) from the graph.
        Action 1: Feature[Retrieve[Strongly Interacting Higgs Sector in the Minimal Standard Model], year]
        Observation 1: The ID of this node is 3101448248. The year is 1993.
        Plan 2: The question is asking the published time of a node (Strongly Interacting Higgs Sector in the Minimal Standard Model). We get the answer: 1993
        Thought 2: The published year of this paper is 1993.
        Action 2: Finish[1993]

        Definition of the graph: There are three types of nodes in the graph: paper, author and venue.
        Paper nodes have features: title, abstract, year and label. Author nodes have features: name. Venue nodes have features: name.
        Paper nodes are linked to their author nodes, venue nodes, reference paper nodes and cited_by paper nodes. Author nodes are linked to their paper nodes. Venue nodes are linked to their paper nodes.
        Question: How many authors do the paper Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars have?
        Plan 1: The question is asking the number of authors of a paper (Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars). Thus, we need to locate the relevant paper and count the authors from the graph.
        Thought 1:  We need to find the node in the graph and calculate the node's author neighbor degree from the graph.
        Action 1: Degree[Retrieve[Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars], author]
        Observation 1: The ID of this node is 2090642949. 2.
        Plan 2: The question is asking the number of authors of a paper (Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars). We get the answer —— 2
        Thought 2: The number of the authors is 2.
        Action 2: Finish[2]

        Definition of the graph: There are three types of nodes in the graph: paper, author and venue.
        Paper nodes have features: title, abstract, year and label. Author nodes have features: name. Venue nodes have features: name.
        Paper nodes are linked to their author nodes, venue nodes, reference paper nodes and cited_by paper nodes. Author nodes are linked to their paper nodes. Venue nodes are linked to their paper nodes.
        Question: What was the publish venue of the paper Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars?
        Plan 1: The question is asking the published venue of a paper (Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars). Thus, we need to locate the relevant paper and get the venue name of the paper from the graph.
        Thought 1: We need to find the node in the graph.
        Action 1: Retrieve[Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars]
        Observation 1: The ID of this node is 2090642949.
        Plan 2: The question is asking the published venue of a paper (Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars). We have the ID of this node. Thus, we need to get the venue name of the paper from the graph.
        Thought 2: We need to check the node's venue neighbor from the graph and get their names.
        Action 2: Feature[Neighbor[2090642949, venue], name]
        Observation 2: ['1980519', '1053242'], the astrophysical journal, the atmosphere journal
        Plan 3: The question is asking the published venue of a paper (Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars). We get the answer: the astrophysical journal, the atmosphere journal
        Thought 3: The name of the published venues are the astrophysical journal and the atmosphere journal.
        Action 3: Finish[the astrophysical journal, the atmosphere journal]
        """,
        "amazon": """Definition of the graph: There are two types of nodes in the graph: item and brand. 
            Item nodes have features: title, description, price, img, category. Brand nodes have features: name. 
            Item nodes are linked to their brand nodes, also_viewed_item nodes, buy_after_viewing_item nodes, also_bought_item nodes, bought_together_item nodes. Brand nodes are linked to their item nodes.
            Question: What is the price of the item Purple Sequin Tiny Dancer Tutu Ballet Dance Fairy Princess Costume Accessory?
            Plan 1: The question is asking the price of a item node (Purple Sequin Tiny Dancer Tutu Ballet Dance Fairy Princess Costume Accessory). Thus, we need to locate the relevant item and extract the price from the graph.
            Thought 1: We need to find the node in the graph and check the node feature (price) from the graph.
            Action 1: Feature[Retrieve[Purple Sequin Tiny Dancer Tutu Ballet Dance Fairy Princess Costume Accessory], price]
            Observation 1: The ID of this node is 0000037214. The price of the item is 6.99.
            Plan 2: The question is asking the price of a item node (Purple Sequin Tiny Dancer Tutu Ballet Dance Fairy Princess Costume Accessory). We get the answer: 6.99.
            Thought 2: The price of the item is 6.99.
            Action 2: Finish[6.99]
            
            Definition of the graph: There are two types of nodes in the graph: item and brand. 
            Item nodes have features: title, description, price, img, category. Brand nodes have features: name. 
            Item nodes are linked to their brand nodes, also_viewed_item nodes, buy_after_viewing_item nodes, also_bought_item nodes, bought_together_item nodes. Brand nodes are linked to their item nodes.
            Question: How many also_viewed_item does item The Sherlock Holmes Audio Collection have?
            Plan 1: The question is asking the number of also_viewed_item neighbour of a item node (The Sherlock Holmes Audio Collection). Thus, We need to find the node in the graph and then count the node's also_viewed_item neighbor.
            Thought 1: We need to find the node in the graph and calculate the node's also_viewed_item neighbor degree from the graph.
            Action 1: Degree[Retrieve[The Sherlock Holmes Audio Collection], also_viewed_item]
            Observation 1: The ID of this node is 0001048236. 9
            Plan 2: The question is asking the number of also_viewed_item neighbour of a item node (The Sherlock Holmes Audio Collection). We get the answer: 9.
            Thought 2: The number of also_viewed_item neighbors is 9.
            Action 2: Finish[9]
            
            Definition of the graph: There are two types of nodes in the graph: item and brand. 
            Item nodes have features: title, description, price, img, category. Brand nodes have features: name. 
            Item nodes are linked to their brand nodes, also_viewed_item nodes, buy_after_viewing_item nodes, also_bought_item nodes, bought_together_item nodes. Brand nodes are linked to their item nodes.
            Question: What is the brand of item Dr. Suess 19163 Dr. Seuss Puzzle 3 Pack Bundle?
            Plan 1: The question is asking brand of a item node (Dr. Suess 19163 Dr. Seuss Puzzle 3 Pack Bundle). We need to find the node and get its brand in the graph.
            Thought 1: We need to find the node in the graph and then check the node's brand neighbor from the graph.
            Action 1: Neighbor[Retrieve[Dr. Suess 19163 Dr. Seuss Puzzle 3 Pack Bundle], brand]
            Observation 1: The ID of this node is 0000191639. ['brand_5']
            Plan 2: The question is asking brand of a item node (Dr. Suess 19163 Dr. Seuss Puzzle 3 Pack Bundle). We get the ID of the brand node. Thus, we need to get the brand name.
            Thought 2: The brand of item is ['brand_5']. The ID of this brand node is brand_5. We need to check the brand node's feature (name) from the graph.
            Action 2: Feature[brand_5, name]
            Observation 2: Dr. Seuss.
            Plan 3: The question is asking brand of a item node (Dr. Suess 19163 Dr. Seuss Puzzle 3 Pack Bundle). We get the answer: Dr. Seuss.
            Thought 3: The brand name of the item is Dr. Seuss.
            Action 3: Finish[Dr. Seuss]
        """,
        "biomedical": """Definition of the graph: There are eleven types of nodes in the graph: Anatomy, Biological Process, Cellular Component, Compound, Disease, Gene, Molecular Function, Pathway, Pharmacologic Class, Side Effect, Symptom.
        Each node has name feature.
        There are these types of edges: Anatomy-downregulates-Gene, Anatomy-expresses-Gene, Anatomy-upregulates-Gene, Compound-binds-Gene, Compound-causes-Side Effect, Compound-downregulates-Gene, Compound-palliates-Disease, Compound-resembles-Compound, Compound-treats-Disease, Compound-upregulates-Gene, Disease-associates-Gene, Disease-downregulates-Gene, Disease-localizes-Anatomy, Disease-presents-Symptom, Disease-resembles-Disease, Disease-upregulates-Gene, Gene-covaries-Gene, Gene-interacts-Gene, Gene-participates-Biological Process, Gene-participates-Cellular Component, Gene-participates-Molecular Function, Gene-participates-Pathway, Gene-regulates-Gene, Pharmacologic Class-includes-Compound.
        Question: What compounds can be used to treat Crohn's disease? Please answer the compound names rather than IDs.
        Plan 1: The question is asking the compounds which can be used to treat a disease (Crohn's disease). We need to find the node in the graph and then find the compounds which can be used to treat this disease.
        Thought 1: We need to find the node in the graph.
        Action 1: Retrieve[Crohn's disease]
        Observation 1: The ID of this node is DOID:8778.
        Plan 2: The question is asking the compounds which can be used to treat a disease (Crohn's disease). We have the ID of this node. Thus, we need to find the compounds which can be used to treat this disease.
        Thought 2: We need to check the node's 'Compound-treats-Disease' neighbor from the graph and get their names.
        Action 2: Feature[Neighbor[DOID:8778, Compound-treats-Disease], name]
        Observation 2: ['DB01014', 'DB00244', 'DB00795', 'DB00993', 'DB00635', 'DB01033'] Balsalazide, Balsalazide, Mesalazine, Sulfasalazine, Azathioprine, Prednisone, Mercaptopurine
        Plan 3: The question is asking the compounds which can be used to treat a disease (Crohn's disease). We get the answer: Balsalazide, Mesalazine, Sulfasalazine, Azathioprine, Prednisone, Mercaptopurine.
        Thought 3: The name of compounds are Balsalazide, Mesalazine, Sulfasalazine, Azathioprine, Prednisone, Mercaptopurine.
        Action 3: Finish[Balsalazide, Mesalazine, Sulfasalazine, Azathioprine, Prednisone, Mercaptopurine]
        
        Definition of the graph: There are eleven types of nodes in the graph: Anatomy, Biological Process, Cellular Component, Compound, Disease, Gene, Molecular Function, Pathway, Pharmacologic Class, Side Effect, Symptom.
        Each node has name feature.
        There are these types of edges: Anatomy-downregulates-Gene, Anatomy-expresses-Gene, Anatomy-upregulates-Gene, Compound-binds-Gene, Compound-causes-Side Effect, Compound-downregulates-Gene, Compound-palliates-Disease, Compound-resembles-Compound, Compound-treats-Disease, Compound-upregulates-Gene, Disease-associates-Gene, Disease-downregulates-Gene, Disease-localizes-Anatomy, Disease-presents-Symptom, Disease-resembles-Disease, Disease-upregulates-Gene, Gene-covaries-Gene, Gene-interacts-Gene, Gene-participates-Biological Process, Gene-participates-Cellular Component, Gene-participates-Molecular Function, Gene-participates-Pathway, Gene-regulates-Gene, Pharmacologic Class-includes-Compound.
        Question: What is the inchikey of Caffeine?
        Plan 1: The question is asking the inchikey of a compound node (Caffeine). We need to find the node in the graph and then get its inchikey.
        Thought 1:We need to find the node in the graph and check the node's 'inchikey' feature from the graph.
        Action 1: Feature[Retrieve[Caffeine], inchikey]
        Observation 1: The ID of this node is DB00201. InChIKey=RYYVLZVUVIJVGH-UHFFFAOYSA-N
        Plan 2: The question is asking the inchikey of a compound node (Caffeine). We get the answer: InChIKey=RYYVLZVUVIJVGH-UHFFFAOYSA-N
        Thought 2:  The inchikey of the node is InChIKey=RYYVLZVUVIJVGH-UHFFFAOYSA-N.
        Action 2: Finish[InChIKey=RYYVLZVUVIJVGH-UHFFFAOYSA-N]
        
        Definition of the graph: There are eleven types of nodes in the graph: Anatomy, Biological Process, Cellular Component, Compound, Disease, Gene, Molecular Function, Pathway, Pharmacologic Class, Side Effect, Symptom.
        Each node has name feature.
        There are these types of edges: Anatomy-downregulates-Gene, Anatomy-expresses-Gene, Anatomy-upregulates-Gene, Compound-binds-Gene, Compound-causes-Side Effect, Compound-downregulates-Gene, Compound-palliates-Disease, Compound-resembles-Compound, Compound-treats-Disease, Compound-upregulates-Gene, Disease-associates-Gene, Disease-downregulates-Gene, Disease-localizes-Anatomy, Disease-presents-Symptom, Disease-resembles-Disease, Disease-upregulates-Gene, Gene-covaries-Gene, Gene-interacts-Gene, Gene-participates-Biological Process, Gene-participates-Cellular Component, Gene-participates-Molecular Function, Gene-participates-Pathway, Gene-regulates-Gene, Pharmacologic Class-includes-Compound.
        Question: How many side effects does Caffeine have?
        Plan 1: The question is asking the number of side effects of a compound node (Caffeine). We need to find the node in the graph and then get the number of the node's Compound-causes-Side Effect.
        Thought 1: The question is related to a compound node (Caffeine). We need to find the node in the graph and calculate the number of the node's 'Compound-causes-Side Effect' neighbors from the graph.
        Action 1: Degree[Retrieve[Caffeine], 'Compound-causes-Side Effect']
        Observation 1: The ID of this node is DB00201. 58
        Plan 2: The question is asking the number of side effects of a compound node (Caffeine). We get the answer: 58.
        Thought 2: TThe number of 'Compound-causes-Side Effect' neighbors are 58.
        Action 2: Finish[58]
        """,
        "legal": """Definition of the graph: There are four types of nodes in the graph: opinion, opinion_cluster, docket, and court.
        Opinion nodes have features: plain_text. Opinion_cluster nodes have features: syllabus, judges, case_name, attorneys. Docket nodes have features: pacer_case_id, case_name. Court nodes have features: full_name, start_date, end_date, citation_string.
        Opinion nodes are linked to their reference nodes and cited_by nodes, as well as their opinion_cluster nodes. Opinion_cluster nodes are linked to opinion nodes and docket nodes. Docket nodes are linked to opinion_cluster nodes and court nodes. Court nodes are linked to docket nodes.
        Question: Who are the judges that involved in the opinion_cluster: <p>Constitutional law — Title of act — Mortgage—Deed—Liability of grantee — Act of June IB, 1878, P. L. BOB.</p> <p>1. The Act of June 12, 1878, P. L. 205, entitled “An Act to define the liability of grantees of real estate for the incumbrances thereon,” is sufficiently comprehensive in its title to cover both the first and second sections of the act, and does not violate art. Ill, sec. 3 of the constitution of Pennsylvania.</p> <p>2. Where an owner of land creates two mortgages thereon and then conveys the property subject to the mortgages, the grantee covenanting in the deed itself to assume and pay the mortgages as a part of the consideration, and thereafter the grantee conveys the land to another person without expressly assuming any continuing liability to pay the mortgages, such grantee is not liable to the mortgag^ for payment of the mortgages. £V</p>?
        Plan 1: The question is asking the judges of this opinion_cluster node (<p>Constitutional law — Title of act — Mortgage—Deed—Liability of grantee — Act of June IB, 1878, P. L. BOB.</p> <p>1. The Act of June 12, 1878, P. L. 205, entitled “An Act to define the liability of grantees of real estate for the incumbrances thereon,” is sufficiently comprehensive in its title to cover both the first and second sections of the act, and does not violate art. Ill, sec. 3 of the constitution of Pennsylvania.</p> <p>2. Where an owner of land creates two mortgages thereon and then conveys the property subject to the mortgages, the grantee covenanting in the deed itself to assume and pay the mortgages as a part of the consideration, and thereafter the grantee conveys the land to another person without expressly assuming any continuing liability to pay the mortgages, such grantee is not liable to the mortgag^ for payment of the mortgages. £V</p>). We need to find the node in the graph and then get the judges.
        Thought 1: We need to find the node in the graph and check the node's 'judges' feature from the graph.
        Action 1: Feature[Retrieve[<p>Constitutional law — Title of act — Mortgage—Deed—Liability of grantee — Act of June IB, 1878, P. L. BOB.</p> <p>1. The Act of June 12, 1878, P. L. 205, entitled “An Act to define the liability of grantees of real estate for the incumbrances thereon,” is sufficiently comprehensive in its title to cover both the first and second sections of the act, and does not violate art. Ill, sec. 3 of the constitution of Pennsylvania.</p> <p>2. Where an owner of land creates two mortgages thereon and then conveys the property subject to the mortgages, the grantee covenanting in the deed itself to assume and pay the mortgages as a part of the consideration, and thereafter the grantee conveys the land to another person without expressly assuming any continuing liability to pay the mortgages, such grantee is not liable to the mortgag^ for payment of the mortgages. £V</p>], judges]
        Observation 1: The ID of this node is opc-6381448.  Brown, Elkin, Fell, Mestrezat, Moschzisker, Potter, Stewart
        Plan 2: The question is asking the judges of this opinion_cluster node (<p>Constitutional law — Title of act — Mortgage—Deed—Liability of grantee — Act of June IB, 1878, P. L. BOB.</p> <p>1. The Act of June 12, 1878, P. L. 205, entitled “An Act to define the liability of grantees of real estate for the incumbrances thereon,” is sufficiently comprehensive in its title to cover both the first and second sections of the act, and does not violate art. Ill, sec. 3 of the constitution of Pennsylvania.</p> <p>2. Where an owner of land creates two mortgages thereon and then conveys the property subject to the mortgages, the grantee covenanting in the deed itself to assume and pay the mortgages as a part of the consideration, and thereafter the grantee conveys the land to another person without expressly assuming any continuing liability to pay the mortgages, such grantee is not liable to the mortgag^ for payment of the mortgages. £V</p>). We get the answer: Brown, Elkin, Fell, Mestrezat, Moschzisker, Potter, Stewart.
        Thought 2: The judges of this opinion cluster is Brown, Elkin, Fell, Mestrezat, Moschzisker, Potter, Stewart
        Action 2: Finish[Brown, Elkin, Fell, Mestrezat, Moschzisker, Potter, Stewart]

        Definition of the graph: There are four types of nodes in the graph: opinion, opinion_cluster, docket, and court.
        Opinion nodes have features: plain_text. Opinion_cluster nodes have features: syllabus, judges, case_name, attorneys. Docket nodes have features: pacer_case_id, case_name. Court nodes have features: full_name, start_date, end_date, citation_string.
        Opinion nodes are linked to their reference nodes and cited_by nodes, as well as their opinion_cluster nodes. Opinion_cluster nodes are linked to opinion nodes and docket nodes. Docket nodes are linked to opinion_cluster nodes and court nodes. Court nodes are linked to docket nodes.
        Question: How many opinions are contained in this opinion_cluster: <p>A general statement that the decision of a state court is against the constitutional rights of the objecting party, or against the Fourteenth Amendment, or that it is without due process of law,-particularly when these objections appear only in specifications of error, so called, will not raise a Federal question, even where the judgment is a final one within Rev. Stat. § 709.</p> <p>In these cases there was no final judgment, such as is provided for in Rev. Stat. § 709, and there does not appear to have arisen any Federal question whatever.</p>'},
        plan 1: The question is asking the number of the opinion neighbors of this node (<p>A general statement that the decision of a state court is against the constitutional rights of the objecting party, or against the Fourteenth Amendment, or that it is without due process of law,-particularly when these objections appear only in specifications of error, so called, will not raise a Federal question, even where the judgment is a final one within Rev. Stat. § 709.</p> <p>In these cases there was no final judgment, such as is provided for in Rev. Stat. § 709, and there does not appear to have arisen any Federal question whatever.</p>). We need to find the node in the graph and count its opinion neighbors.
        Thought 1: We need to find the node in the graph and calculate the node's "opinion" neighbor degree from the graph.
        Action 1: Degree[Retrieve[<p>A general statement that the decision of a state court is against the constitutional rights of the objecting party, or against the Fourteenth Amendment, or that it is without due process of law,-particularly when these objections appear only in specifications of error, so called, will not raise a Federal question, even where the judgment is a final one within Rev. Stat. § 709.</p> <p>In these cases there was no final judgment, such as is provided for in Rev. Stat. § 709, and there does not appear to have arisen any Federal question whatever.</p>], opinion]
        Observation 1: The ID of this node is opc-94592. 1
        Plan 2: The question is asking the number of the opinion neighbors of this node (<p>A general statement that the decision of a state court is against the constitutional rights of the objecting party, or against the Fourteenth Amendment, or that it is without due process of law,-particularly when these objections appear only in specifications of error, so called, will not raise a Federal question, even where the judgment is a final one within Rev. Stat. § 709.</p> <p>In these cases there was no final judgment, such as is provided for in Rev. Stat. § 709, and there does not appear to have arisen any Federal question whatever.</p>). We get the answer: 1
        Thought 2: The number of the opinion neighbors is 1
        Action 2: Finish[1]

        Definition of the graph: There are four types of nodes in the graph: opinion, opinion_cluster, docket, and court.
        Opinion nodes have features: plain_text. Opinion_cluster nodes have features: syllabus, judges, case_name, attorneys. Docket nodes have features: pacer_case_id, case_name. Court nodes have features: full_name, start_date, end_date, citation_string.
        Opinion nodes are linked to their reference nodes and cited_by nodes, as well as their opinion_cluster nodes. Opinion_cluster nodes are linked to opinion nodes and docket nodes. Docket nodes are linked to opinion_cluster nodes and court nodes. Court nodes are linked to docket nodes.
        Question: What is the docket pacer case ID for this opinion cluster: In May, 1873, a brewer pays to the collector $100 as special tax, for which a special-tax stamp is issued to Mm. Subsequently he shoivs to. the satisfaction of the Commissions)\' of Internal Revenue that he was not a brewer of more than 500 barrels, and henee that he is entitled to have $50 refunded. The Commissioner issues his certificate for a special-tax stamp of the value of%50 returned. The Treasury refuses payment of the certificate. The breioe)\' brings his action to recover the amount thereof.</p> <p>I. The Commissioner of Internal Revenue being authorized by one section of the statutes (Rev. Stat., § 3220) to refund taxes “unjustly assessed or excessive in, amount," and by another, (§ 3426,) the value of special-tax stamps ‘‘where the rates of duties represented thereby have been paid in error or remitted," it is immaterial whether his certificate be in form under the one section or the other so long as it represents a case where the Commissioner is authorized to refund and the amoirnt which the party is entitled to have refunded.</p> <p>II. When a controversy under the revenue laws is by law required to be determined by other officers and other tribunals, this court is without jurisdiction; but where the proper officer has determined a question, and awarded an allowance, and filed a certificate, and exhausted his jurisdiction, and the Treasury has refused to carry out the award by mating payment, this court has jurisdiction of an action brought to recover the. award..
        Plan 1: The question is asking the judges of this opinion cluster node (In May, 1873, a brewer pays to the collector $100 as special tax, for which a special-tax stamp is issued to Mm. Subsequently he shoivs to. the satisfaction of the Commissions)\' of Internal Revenue that he was not a brewer of more than 500 barrels, and henee that he is entitled to have $50 refunded. The Commissioner issues his certificate for a special-tax stamp of the value of%50 returned. The Treasury refuses payment of the certificate. The breioe)\' brings his action to recover the amount thereof.</p> <p>I. The Commissioner of Internal Revenue being authorized by one section of the statutes (Rev. Stat., § 3220) to refund taxes “unjustly assessed or excessive in, amount," and by another, (§ 3426,) the value of special-tax stamps ‘‘where the rates of duties represented thereby have been paid in error or remitted," it is immaterial whether his certificate be in form under the one section or the other so long as it represents a case where the Commissioner is authorized to refund and the amoirnt which the party is entitled to have refunded.</p> <p>II. When a controversy under the revenue laws is by law required to be determined by other officers and other tribunals, this court is without jurisdiction; but where the proper officer has determined a question, and awarded an allowance, and filed a certificate, and exhausted his jurisdiction, and the Treasury has refused to carry out the award by mating payment, this court has jurisdiction of an action brought to recover the. award.). We need to find the node in the graph and get its judges.
        Thought 1: We need to find the node in the graph and check the node's 'judges' feature from the graph.
        Action 1: Feature[Retrieve[In May, 1873, a brewer pays to the collector $100 as special tax, for which a special-tax stamp is issued to Mm. Subsequently he shoivs to. the satisfaction of the Commissions)\' of Internal Revenue that he was not a brewer of more than 500 barrels, and henee that he is entitled to have $50 refunded. The Commissioner issues his certificate for a special-tax stamp of the value of%50 returned. The Treasury refuses payment of the certificate. The breioe)\' brings his action to recover the amount thereof.</p> <p>I. The Commissioner of Internal Revenue being authorized by one section of the statutes (Rev. Stat., § 3220) to refund taxes “unjustly assessed or excessive in, amount," and by another, (§ 3426,) the value of special-tax stamps ‘‘where the rates of duties represented thereby have been paid in error or remitted," it is immaterial whether his certificate be in form under the one section or the other so long as it represents a case where the Commissioner is authorized to refund and the amoirnt which the party is entitled to have refunded.</p> <p>II. When a controversy under the revenue laws is by law required to be determined by other officers and other tribunals, this court is without jurisdiction; but where the proper officer has determined a question, and awarded an allowance, and filed a certificate, and exhausted his jurisdiction, and the Treasury has refused to carry out the award by mating payment, this court has jurisdiction of an action brought to recover the. award.]]
        Observation 1: The ID of this node is opc-8599951. {'judges': 'Eichaedson', 'case_name': 'Kaufman v. United States', 'attorneys': 'Mr. P. P. Pye for the claimant., Mr. Assistant Attorney-General Simons for the defendants.', 'syllabus': '<p>In May, 1873, a brewer pays to the collector $100 as special tax, for which a special-tax stamp is issued to Mm. Subsequently he shoivs to. the satisfaction of the Commissions)\' of Internal Revenue that he was not a brewer of more than 500 barrels, and henee that he is entitled to have $50 refunded. The Commissioner issues his certificate for a special-tax stamp of the value of%50 returned. The Treasury refuses payment of the certificate. The breioe)\' brings his action to recover the amount thereof.</p> <p>I. The Commissioner of Internal Revenue being authorized by one section of the statutes (Rev. Stat., § 3220) to refund taxes “unjustly assessed or excessive in, amount," and by another, (§ 3426,) the value of special-tax stamps ‘‘where the rates of duties represented thereby have been paid in error or remitted," it is immaterial whether his certificate be in form under the one section or the other so long as it represents a case where the Commissioner is authorized to refund and the amoirnt which the party is entitled to have refunded.</p> <p>II. When a controversy under the revenue laws is by law required to be determined by other officers and other tribunals, this court is without jurisdiction; but where the proper officer has determined a question, and awarded an allowance, and filed a certificate, and exhausted his jurisdiction, and the Treasury has refused to carry out the award by mating payment, this court has jurisdiction of an action brought to recover the. award.</p>'}
        Plan 2: The question is asking the judges of this opinion cluster node (In May, 1873, a brewer pays to the collector $100 as special tax, for which a special-tax stamp is issued to Mm. Subsequently he shoivs to. the satisfaction of the Commissions)\' of Internal Revenue that he was not a brewer of more than 500 barrels, and henee that he is entitled to have $50 refunded. The Commissioner issues his certificate for a special-tax stamp of the value of%50 returned. The Treasury refuses payment of the certificate. The breioe)\' brings his action to recover the amount thereof.</p> <p>I. The Commissioner of Internal Revenue being authorized by one section of the statutes (Rev. Stat., § 3220) to refund taxes “unjustly assessed or excessive in, amount," and by another, (§ 3426,) the value of special-tax stamps ‘‘where the rates of duties represented thereby have been paid in error or remitted," it is immaterial whether his certificate be in form under the one section or the other so long as it represents a case where the Commissioner is authorized to refund and the amoirnt which the party is entitled to have refunded.</p> <p>II. When a controversy under the revenue laws is by law required to be determined by other officers and other tribunals, this court is without jurisdiction; but where the proper officer has determined a question, and awarded an allowance, and filed a certificate, and exhausted his jurisdiction, and the Treasury has refused to carry out the award by mating payment, this court has jurisdiction of an action brought to recover the. award.). We get the answer: Eichaedson.
        Thought 2: The judges of this opinion cluster is Eichaedson.
        Action 2: Finish[Eichaedson]
        """,
    "goodreads": """Definition of the graph: There are four types of nodes in the graph: book, author, publisher, and series. 
        Book nodes have features: country_code, language_code, is_ebook, title, description, format, num_pages, publication_year, url, popular_shelves, and genres. Author nodes have features: name. Publisher nodes have features: name. Series nodes have features: title and description. 
        Book nodes are linked to their author nodes, publisher nodes, series nodes and similar_books nodes. Author nodes are linked to their book nodes. Publisher nodes are linked to their book nodes. Series nodes are linked to their book nodes.
        Question: Who is the author of the book 'The Great Gatsby'?
        Plan 1: The question is asking for the author of a specific book (The Great Gatsby). We need to find the book node and then get its author.
        Thought 1: We need to find the book node in the graph and check the book node's author neighbor from the graph.
        Action 1: Neighbor[Retrieve[The Great Gatsby], author]
        Observation 1: The ID of this book node is 21366268. ['3190']
        Plan 2: The question is asking for the author of a specific book (The Great Gatsby). We have the author's ID. Thus, we need to get the author's name.
        Thought 2: The ID of the author is 3190. We need to check the name.
        Action 2: Feature[3190, name]
        Observation 2: F. Scott Fitzgerald
        Plan 3: The question is asking for the author of a specific book (The Great Gatsby). We have the answer: F. Scott Fitzgerald.
        Thought 3: The author of 'The Great Gatsby' is F. Scott Fitzgerald.
        Action 3: Finish[F. Scott Fitzgerald]
        
        Definition of the graph: There are four types of nodes in the graph: book, author, publisher, and series. 
        Book nodes have features: country_code, language_code, is_ebook, title, description, format, num_pages, publication_year, url, popular_shelves, and genres. Author nodes have features: name. Publisher nodes have features: name. Series nodes have features: title and description. 
        Book nodes are linked to their author nodes, publisher nodes, series nodes and similar_books nodes. Author nodes are linked to their book nodes. Publisher nodes are linked to their book nodes. Series nodes are linked to their book nodes.
        Question: What is the genre of the book '1984'?
        Plan 1: The question is asking for the genre of a specific book (1984). We need to find the book node and then get its genre.
        Thought 1: We need to find the book node in the graph and then check the book node's genre feature from the graph.
        Action 1: Feature[Retrieve[1984], genres]
        Observation 1: The ID of this book node is 3475269. currently-reading, classics, fiction
        Plan 2: The question is asking for the genre of a specific book (1984). We have the answer: currently-reading, classics, fiction.
        Thought 2: The genres of the book '1984' are currently-reading, classics, fiction.
        Action 2: Finish[currently-reading, classics, fiction]
        
        Definition of the graph: There are four types of nodes in the graph: book, author, publisher, and series. 
        Book nodes have features: country_code, language_code, is_ebook, title, description, format, num_pages, publication_year, url, popular_shelves, and genres. Author nodes have features: name. Publisher nodes have features: name. Series nodes have features: title and description. 
        Book nodes are linked to their author nodes, publisher nodes, series nodes and similar_books nodes. Author nodes are linked to their book nodes. Publisher nodes are linked to their book nodes. Series nodes are linked to their book nodes.
        Question: How many series included the book 'Harry Potter and the Sorcerer's Stone'?
        Plan 1: The question is asking for the number of series included in a specific book (Harry Potter and the Sorcerer's Stone). We need to find the book node and then get its series neighbor degree.
        Thought 1: We need to find the book node for 'Harry Potter and the Sorcerer's Stone' to determine its series and calculate the node's series neighbor degree from the graph.
        Action 1: Degree[Retrieve[Harry Potter and the Sorcerer's Stone], series]
        Observation 1: The ID of this book node is 27421523. 1
        Plan 2: The question is asking for the number of series included in a specific book (Harry Potter and the Sorcerer's Stone). We have the answer: 1.
        Thought 2: The number of series neighbors is 1.
        Action 2: Finish[1]
        """,
    "dblp": """Definition of the graph: There are three types of nodes in the graph: paper, author and venue.
        Paper nodes have features: title, abstract, keywords, lang, and year. Author nodes have features: name and organization. Venue nodes have features: name.
        Paper nodes are linked to their author nodes, venue nodes, reference nodes (the papers this paper cite) and cited_by nodes (other papers which cite this paper). Author nodes are linked to their paper nodes. Venue nodes are linked to their paper nodes.
        Question: What organization is researcher Greg Daville from?
        Plan 1: The question is asking about the organization of a researcher named Greg Daville. We need to find the node of Greg Daville in the graph and check the node's organization feature from the graph.
        Thought 1: We need to find the node corresponding to Greg Daville in the graph and check the node's organization feature from the graph.
        Action 1: Feature[Retrieve[Greg Daville], organization]
        Observation 1: The ID of this retrieval target node is 53f460a7dabfaee4dc83702a. Hove East Sussex, United Kingdom
        Plan 2: The question is asking about the organization of a researcher named Greg Daville. We have the answer: Hove East Sussex, United Kingdom
        Thought 2: The organization of researcher Greg Daville is Hove East Sussex, United Kingdom.
        Action 2: Finish[Hove East Sussex, United Kingdom]
    
        Definition of the graph: There are three types of nodes in the graph: paper, author and venue.
        Paper nodes have features: title, abstract, keywords, lang, and year. Author nodes have features: name and organization. Venue nodes have features: name.
        Paper nodes are linked to their author nodes, venue nodes, reference nodes (the papers this paper cite) and cited_by nodes (other papers which cite this paper). Author nodes are linked to their paper nodes. Venue nodes are linked to their paper nodes.
        Question: How many papers are written by author Nicholas Lydon?
        Plan 1: The question is asking for the number of papers written by a specific author (Nicholas Lydon). We need to find the author node in the graph and then calculate the number of the author's papers.
        Thought 1: We need to find the author node in the graph and calculate the "paper" neighbor degree of this node.
        Action 1: Degree[Retrieve[Nicholas Lydon], paper]
        Observation 1: The ID of this retrieval target node is 53f438c3dabfaedf43596117. 2
        Plan 2: The question is asking for the number of papers written by a specific author (Nicholas Lydon). We have the answer: 2
        Thought 2: The number of the paper neighbors is 2
        Action 2: Finish[2]

        Definition of the graph: There are three types of nodes in the graph: paper, author and venue.
        Paper nodes have features: title, abstract, keywords, lang, and year. Author nodes have features: name and organization. Venue nodes have features: name.
        Paper nodes are linked to their author nodes, venue nodes, reference nodes (the papers this paper cite) and cited_by nodes (other papers which cite this paper). Author nodes are linked to their paper nodes. Venue nodes are linked to their paper nodes.
        Question: What was the publish venue of the paper Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars?
        Plan 1: The question is asking the published venue of a paper (Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars). We need to find the paper node and then get its venue.
        Thought 1: We need to find the node in the graph.
        Action 1: Retrieve[Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars]
        Observation 1: The ID of this node is 2090642949.
        Plan 2: The question is asking the published venue of a paper (Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars). We have the ID of the paper node. Thus, we need to get the venue's name.
        Thought 2: We need to check the node's venue neighbor from the graph and get their names.
        Action 2: Feature[Neighbor[2090642949, venue], name]
        Observation 2: ['1980519', '1053242'] the astrophysical journal, the atmosphere journal
        Plan 3: The question is asking the published venue of a paper (Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars). We have the answer: the astrophysical journal, the atmosphere journal
        Thought 3: The name of the published venues are the astrophysical journal and the atmosphere journal
        Action 3: Finish[the astrophysical journal, the atmosphere journal]
        """,
}

PLAN_EXAMPLES = {
    "maple": """Definition of the graph: There are three types of nodes in the graph: paper, author and venue.
        Paper nodes have features: title, abstract, year and label. Author nodes have features: name. Venue nodes have features: name.
        Paper nodes are linked to their author nodes, venue nodes, reference paper nodes and cited_by paper nodes. Author nodes are linked to their paper nodes. Venue nodes are linked to their paper nodes.
        Question: When was the paper Strongly Interacting Higgs Sector in the Minimal Standard Model published?
        Plan 1: The question is asking the published time of a node (Strongly Interacting Higgs Sector in the Minimal Standard Model). Thus, we need to locate the relevant paper and extract the published year from the graph.
        Thought 1: We need to find the node in the graph and check the node feature (year) from the graph.
        Action 1: Feature[Retrieve[Strongly Interacting Higgs Sector in the Minimal Standard Model], year]
        Observation 1: The ID of this node is 3101448248. The year is 1993.
        Plan 2: The question is asking the published time of a node (Strongly Interacting Higgs Sector in the Minimal Standard Model). We have got the answer is 1993.
        Thought 2: The published year of this paper is 1993.
        Action 2: Finish[1993]

        Definition of the graph: There are three types of nodes in the graph: paper, author and venue.
        Paper nodes have features: title, abstract, year and label. Author nodes have features: name. Venue nodes have features: name.
        Paper nodes are linked to their author nodes, venue nodes, reference paper nodes and cited_by paper nodes. Author nodes are linked to their paper nodes. Venue nodes are linked to their paper nodes.
        Question: How many authors do the paper Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars have?
        Plan 1: The question is asking the number of authors of a paper (Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars). Thus, we need to locate the relevant paper and count the authors from the graph.
        Thought 1:  We need to find the node in the graph and calculate the node's author neighbor degree from the graph.
        Action 1: Degree[Retrieve[Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars], author]
        Observation 1: The ID of this node is 2090642949. 2.
        Plan 2: The question is asking the number of authors of a paper (Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars). We have got the answer is 2.
        Thought 2: The number of the authors is 2.
        Action 2: Finish[2]

        Definition of the graph: There are three types of nodes in the graph: paper, author and venue.
        Paper nodes have features: title, abstract, year and label. Author nodes have features: name. Venue nodes have features: name.
        Paper nodes are linked to their author nodes, venue nodes, reference paper nodes and cited_by paper nodes. Author nodes are linked to their paper nodes. Venue nodes are linked to their paper nodes.
        Question: What was the publish venue of the paper Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars?
        Plan 1: The question is asking the published venue of a paper (Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars). Thus, we need to locate the relevant paper and get the venue name of the paper from the graph.
        Thought 1: We need to find the node in the graph.
        Action 1: Retrieve[Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars]
        Observation 1: The ID of this node is 2090642949.
        Plan 2: The question is asking the published venue of a paper (Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars). We have the ID of this node. Thus, we need to get the venue name of the paper from the graph.
        Thought 2: We need to check the node's venue neighbor from the graph and get their names.
        Action 2: Feature[Neighbor[2090642949, venue], name]
        Observation 2: ['1980519', '1053242'], the astrophysical journal, the atmosphere journal
        Plan 3: The question is asking the published venue of a paper (Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars). We have got the answer is the astrophysical journal, the atmosphere journal.
        Thought 3: The name of the published venues are the astrophysical journal and the atmosphere journal.
        Action 3: Finish[the astrophysical journal, the atmosphere journal]

        Definition of the graph: There are three types of nodes in the graph: paper, author and venue.
        Paper nodes have features: title, abstract, year and label. Author nodes have features: name. Venue nodes have features: name.
        Paper nodes are linked to their author nodes, venue nodes, reference paper nodes and cited_by paper nodes. Author nodes are linked to their paper nodes. Venue nodes are linked to their paper nodes.
        Question: When was the paper Strongly Interacting Higgs Sector in the Minimal Standard Model published?
        Plan 1: The question is asking the published time of a node (Strongly Interacting Higgs Sector in the Minimal Standard Model). Thus, we need to locate the relevant paper and extract the published year from the graph.
        Thought 1: We need to find the node in the graph.
        Action 1: Retrieve[Strongly Interacting Higgs Sector in the Minimal Standard Model]
        Observation 1: The ID of this node is 3101448248.
        Plan 2: The question is asking the published time of a node (Strongly Interacting Higgs Sector in the Minimal Standard Model). We have the paper node. Thus, we need to extract the published year from the graph.
        Thought 2: The question is asking the published date of a paper, we need to check the node feature (year) from the graph.
        Action 2: Feature[3101448248, year]
        Observation 2: 1993
        Plan 3: The question is asking the published time of a node (Strongly Interacting Higgs Sector in the Minimal Standard Model). We have the answer ———— 1993.
        Thought 3: The published date of the paper is 1993.
        Action 3: Finish[1993]

        Definition of the graph: There are three types of nodes in the graph: paper, author and venue.
        Paper nodes have features: title, abstract, year and label. Author nodes have features: name. Venue nodes have features: name.
        Paper nodes are linked to their author nodes, venue nodes, reference paper nodes and cited_by paper nodes. Author nodes are linked to their paper nodes. Venue nodes are linked to their paper nodes.
        Question: How many authors do the paper Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars have?
        Plan 1: The question is asking the number of authors of a paper (Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars). Thus, we need to locate the relevant paper and count the authors from the graph.
        Thought 1: We need to find the node in the graph.
        Action 1: Retrieve[Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars]
        Observation 1: The ID of this node is 2090642949.
        Plan 2: The question is asking the number of authors of a paper (Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars). We have the paper node. Thus, we need to count the authors from the graph.
        Thought 2: We need to calculate the node's author neighbor degree from the graph.
        Action 2: Degree[2090642949, author]
        Observation 2: 2
        Plan 3: The question is asking the number of authors of a paper (Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars). We get the answer ———— 2.
        Thought 3: The number of the authors is 2
        Action 3: Finish[2]

        Definition of the graph: There are three types of nodes in the graph: paper, author and venue.
        Paper nodes have features: title, abstract, year and label. Author nodes have features: name. Venue nodes have features: name.
        Paper nodes are linked to their author nodes, venue nodes, reference paper nodes and cited_by paper nodes. Author nodes are linked to their paper nodes. Venue nodes are linked to their paper nodes.
        Question: What was the publish venue of the paper Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars?
        Plan 1: The question is asking information of a node (Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars). Thus, we need to locate the relevant paper and get the venue name of the paper from the graph.
        Thought 1: We need to find the node in the graph.
        Action 1: Retrieve[Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars]
        Observation 1: The ID of this node is 2090642949.
        Plan 2: The question is asking information of a node (Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars). We have the paper node. Thus, we need to get venue neighbor of the paper.
        Thought 2: We need to check the node's venue neighbor from the graph.
        Action 2: Neighbor[2090642949, venue]
        Observation 2: ['1980519', '1053242']
        Plan 3: The question is asking information of a node (Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars). We have the venues node. Thus, we need to get the names of these venues.
        Thought 3: The ID of the published venue are 1980519 and 1053242. We need to get their names.
        Action 3: Feature[1980519, name], Feature[1053242, name]
        Observation 3: the astrophysical journal, the atmosphere journal
        Plan 4: The question is asking the published venue of a paper (Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars). We get the answer ———— the astrophysical journal, the atmosphere journal
        Thought 4: The name of the published venues are the astrophysical journal and the atmosphere journal
        Action 4: Finish[the astrophysical journal, the atmosphere journal]
        """,
        "amazon": """Definition of the graph: There are two types of nodes in the graph: item and brand. 
            Item nodes have features: title, description, price, img, category. Brand nodes have features: name. 
            Item nodes are linked to their brand nodes, also_viewed_item nodes, buy_after_viewing_item nodes, also_bought_item nodes, bought_together_item nodes. Brand nodes are linked to their item nodes.
            Question: What is the price of the item Purple Sequin Tiny Dancer Tutu Ballet Dance Fairy Princess Costume Accessory?
            Plan 1: The question is asking the price of a item node (Purple Sequin Tiny Dancer Tutu Ballet Dance Fairy Princess Costume Accessory). Thus, we need to locate the relevant item and extract the price from the graph.
            Thought 1: We need to find the node in the graph and check the node feature (price) from the graph.
            Action 1: Feature[Retrieve[Purple Sequin Tiny Dancer Tutu Ballet Dance Fairy Princess Costume Accessory], price]
            Observation 1: The ID of this node is 0000037214. The price of the item is 6.99.
            Plan 2: The question is asking the price of a item node (Purple Sequin Tiny Dancer Tutu Ballet Dance Fairy Princess Costume Accessory). We have got the answer is 6.99.
            Thought 2: The price of the item is 6.99.
            Action 2: Finish[6.99]
            
            Definition of the graph: There are two types of nodes in the graph: item and brand. 
            Item nodes have features: title, description, price, img, category. Brand nodes have features: name. 
            Item nodes are linked to their brand nodes, also_viewed_item nodes, buy_after_viewing_item nodes, also_bought_item nodes, bought_together_item nodes. Brand nodes are linked to their item nodes.
            Question: How many also_viewed_item does item The Sherlock Holmes Audio Collection have?
            Plan 1: The question is asking the number of also_viewed_item neighbour of a item node (The Sherlock Holmes Audio Collection). Thus, We need to find the node in the graph and then count the node's also_viewed_item neighbor.
            Thought 1: We need to find the node in the graph and calculate the node's also_viewed_item neighbor degree from the graph.
            Action 1: Degree[Retrieve[The Sherlock Holmes Audio Collection], also_viewed_item]
            Observation 1: The ID of this node is 0001048236. 9
            Plan 2: The question is asking the number of also_viewed_item neighbour of a item node (The Sherlock Holmes Audio Collection). We have got the answer is 9.
            Thought 2: The number of also_viewed_item neighbors is 9.
            Action 2: Finish[9]
            
            Definition of the graph: There are two types of nodes in the graph: item and brand. 
            Item nodes have features: title, description, price, img, category. Brand nodes have features: name. 
            Item nodes are linked to their brand nodes, also_viewed_item nodes, buy_after_viewing_item nodes, also_bought_item nodes, bought_together_item nodes. Brand nodes are linked to their item nodes.
            Question: What is the brand of item Dr. Suess 19163 Dr. Seuss Puzzle 3 Pack Bundle?
            Plan 1: The question is asking brand of a item node (Dr. Suess 19163 Dr. Seuss Puzzle 3 Pack Bundle). We need to find the node and get its brand in the graph.
            Thought 1: We need to find the node in the graph and then check the node's brand neighbor from the graph.
            Action 1: Neighbor[Retrieve[Dr. Suess 19163 Dr. Seuss Puzzle 3 Pack Bundle], brand]
            Observation 1: The ID of this node is 0000191639. ['brand_5']
            Plan 2: The question is asking brand of a item node (Dr. Suess 19163 Dr. Seuss Puzzle 3 Pack Bundle). We get the ID of the brand node. Thus, we need to get the brand name.
            Thought 2: The brand of item is ['brand_5']. The ID of this brand node is brand_5. We need to check the brand node's feature (name) from the graph.
            Action 2: Feature[brand_5, name]
            Observation 2: Dr. Seuss.
            Plan 3: The question is asking brand of a item node (Dr. Suess 19163 Dr. Seuss Puzzle 3 Pack Bundle). We have got the answer is Dr. Seuss.
            Thought 3: The brand name of the item is Dr. Seuss.
            Action 3: Finish[Dr. Seuss]

            Definition of the graph: There are two types of nodes in the graph: item and brand. 
            Item nodes have features: title, description, price, img, category. Brand nodes have features: name. 
            Item nodes are linked to their brand nodes, also_viewed_item nodes, buy_after_viewing_item nodes, also_bought_item nodes, bought_together_item nodes. Brand nodes are linked to their item nodes.
            Question: What is the price of the item Purple Sequin Tiny Dancer Tutu Ballet Dance Fairy Princess Costume Accessory?
            Plan 1: The question is asking the price of a item node (Purple Sequin Tiny Dancer Tutu Ballet Dance Fairy Princess Costume Accessory). Thus, we need to locate the relevant item and extract the price from the graph.
            Thought 1: We need to find the node in the graph.
            Action 1: Retrieve[Purple Sequin Tiny Dancer Tutu Ballet Dance Fairy Princess Costume Accessory]
            Observation 1: The ID of this node is 0000037214.
            Plan 2: The question is asking the price of a item node (Purple Sequin Tiny Dancer Tutu Ballet Dance Fairy Princess Costume Accessory). We have the item node. Thus, we need to extract the price from the graph.
            Thought 2: We need to check the node feature (price) from the graph.
            Action 2: Feature[0000037214, price]
            Observation 2: 6.99
            Plan 3: The question is asking the price of a item node (Purple Sequin Tiny Dancer Tutu Ballet Dance Fairy Princess Costume Accessory). We get the answer ———— 6.99.
            Thought 3: The price of the item is 6.99.
            Action 3: Finish[6.99]
            
            Definition of the graph: There are two types of nodes in the graph: item and brand. 
            Item nodes have features: title, description, price, img, category. Brand nodes have features: name. 
            Item nodes are linked to their brand nodes, also_viewed_item nodes, buy_after_viewing_item nodes, also_bought_item nodes, bought_together_item nodes. Brand nodes are linked to their item nodes.
            Question: How many also_viewed_item does item The Sherlock Holmes Audio Collection have?
            Plan 1: The question is asking the number of also_viewed_item neighbour of a item node (The Sherlock Holmes Audio Collection). Thus, We need to find the node in the graph and then count the node's also_viewed_item neighbor.
            Thought 1: We need to find the node in the graph.
            Action 1: Retrieve[The Sherlock Holmes Audio Collection]
            Observation 1: The ID of this node is 0001048236.
            Plan 2: The question is asking the number of also_viewed_item neighbour of a item node (The Sherlock Holmes Audio Collection). We have the item node. Thus, We need to count the node's also_viewed_item neighbor.
            Thought 2: We need to calculate the node's also_viewed_item neighbor degree from the graph.
            Action 2: Degree[0001048236, also_viewed_item]
            Observation 2: 9
            Plan 3: The question is asking the number of also_viewed_item neighbour of a item node (The Sherlock Holmes Audio Collection). We get the answer ———— 9.
            Thought 3: The number of also_viewed_item neighbors is 9.
            Action 3: Finish[9]
            
            Definition of the graph: There are two types of nodes in the graph: item and brand. 
            Item nodes have features: title, description, price, img, category. Brand nodes have features: name. 
            Item nodes are linked to their brand nodes, also_viewed_item nodes, buy_after_viewing_item nodes, also_bought_item nodes, bought_together_item nodes. Brand nodes are linked to their item nodes.
            Question: What is the brand of item Dr. Suess 19163 Dr. Seuss Puzzle 3 Pack Bundle?
            Plan 1: The question is asking brand of a item node (Dr. Suess 19163 Dr. Seuss Puzzle 3 Pack Bundle). We need to find the node and get its brand in the graph.
            Thought 1: We need to find the node in the graph.
            Action 1: Retrieve[Dr. Suess 19163 Dr. Seuss Puzzle 3 Pack Bundle]
            Observation 1: The ID of this node is 0000191639.
            Plan 2: The question is asking brand of a item node (Dr. Suess 19163 Dr. Seuss Puzzle 3 Pack Bundle). We have the item node. Thus, we need to get its brand in the graph.
            Thought 2: We need to check the node's brand neighbor from the graph.
            Action 2: Neighbor[0000191639, brand]
            Observation 2: ['brand_5']
            Plan 3: The question is asking brand of a item node (Dr. Suess 19163 Dr. Seuss Puzzle 3 Pack Bundle). We have the brand node. Thus we need to get the name of the brand.
            Thought 3: The ID of this brand node is brand_5. We need to check the brand node's feature (name) from the graph.
            Action 3: Feature[brand_5, name]
            Observation 3: Dr. Seuss.
            Plan 4: The question is asking brand of a item node (Dr. Suess 19163 Dr. Seuss Puzzle 3 Pack Bundle). We get the answer ———— Dr. Seuss.
            Thought 4: The brand name of the item is Dr. Seuss.
            Action 4: Finish[Dr. Seuss]
        """,
        "biomedical": """Definition of the graph: There are eleven types of nodes in the graph: Anatomy, Biological Process, Cellular Component, Compound, Disease, Gene, Molecular Function, Pathway, Pharmacologic Class, Side Effect, Symptom.
        Each node has name feature.
        There are these types of edges: Anatomy-downregulates-Gene, Anatomy-expresses-Gene, Anatomy-upregulates-Gene, Compound-binds-Gene, Compound-causes-Side Effect, Compound-downregulates-Gene, Compound-palliates-Disease, Compound-resembles-Compound, Compound-treats-Disease, Compound-upregulates-Gene, Disease-associates-Gene, Disease-downregulates-Gene, Disease-localizes-Anatomy, Disease-presents-Symptom, Disease-resembles-Disease, Disease-upregulates-Gene, Gene-covaries-Gene, Gene-interacts-Gene, Gene-participates-Biological Process, Gene-participates-Cellular Component, Gene-participates-Molecular Function, Gene-participates-Pathway, Gene-regulates-Gene, Pharmacologic Class-includes-Compound.
        Question: What compounds can be used to treat Crohn's disease? Please answer the compound names rather than IDs.
        Plan 1: The question is asking the compounds which can be used to treat a disease (Crohn's disease). We need to find the node in the graph and then find the compounds which can be used to treat this disease.
        Thought 1: We need to find the node in the graph.
        Action 1: Retrieve[Crohn's disease]
        Observation 1: The ID of this node is DOID:8778.
        Plan 2: The question is asking the compounds which can be used to treat a disease (Crohn's disease). We have the ID of this node. Thus, we need to find the compounds which can be used to treat this disease.
        Thought 2: We need to check the node's 'Compound-treats-Disease' neighbor from the graph and get their names.
        Action 2: Feature[Neighbor[DOID:8778, Compound-treats-Disease], name]
        Observation 2: ['DB01014', 'DB00244', 'DB00795', 'DB00993', 'DB00635', 'DB01033'] Balsalazide, Balsalazide, Mesalazine, Sulfasalazine, Azathioprine, Prednisone, Mercaptopurine
        Plan 3: The question is asking the compounds which can be used to treat a disease (Crohn's disease). We have got the answer is Balsalazide, Mesalazine, Sulfasalazine, Azathioprine, Prednisone, Mercaptopurine.
        Thought 3: The name of compounds are Balsalazide, Mesalazine, Sulfasalazine, Azathioprine, Prednisone, Mercaptopurine.
        Action 3: Finish[Balsalazide, Mesalazine, Sulfasalazine, Azathioprine, Prednisone, Mercaptopurine]
        
        Definition of the graph: There are eleven types of nodes in the graph: Anatomy, Biological Process, Cellular Component, Compound, Disease, Gene, Molecular Function, Pathway, Pharmacologic Class, Side Effect, Symptom.
        Each node has name feature.
        There are these types of edges: Anatomy-downregulates-Gene, Anatomy-expresses-Gene, Anatomy-upregulates-Gene, Compound-binds-Gene, Compound-causes-Side Effect, Compound-downregulates-Gene, Compound-palliates-Disease, Compound-resembles-Compound, Compound-treats-Disease, Compound-upregulates-Gene, Disease-associates-Gene, Disease-downregulates-Gene, Disease-localizes-Anatomy, Disease-presents-Symptom, Disease-resembles-Disease, Disease-upregulates-Gene, Gene-covaries-Gene, Gene-interacts-Gene, Gene-participates-Biological Process, Gene-participates-Cellular Component, Gene-participates-Molecular Function, Gene-participates-Pathway, Gene-regulates-Gene, Pharmacologic Class-includes-Compound.
        Question: What is the inchikey of Caffeine?
        Plan 1: The question is asking the inchikey of a compound node (Caffeine). We need to find the node in the graph and then get its inchikey.
        Thought 1:We need to find the node in the graph and check the node's 'inchikey' feature from the graph.
        Action 1: Feature[Retrieve[Caffeine], inchikey]
        Observation 1: The ID of this node is DB00201. InChIKey=RYYVLZVUVIJVGH-UHFFFAOYSA-N
        Plan 2: The question is asking the inchikey of a compound node (Caffeine). We have got the answer is InChIKey=RYYVLZVUVIJVGH-UHFFFAOYSA-N.
        Thought 2:  The inchikey of the node is InChIKey=RYYVLZVUVIJVGH-UHFFFAOYSA-N.
        Action 2: Finish[InChIKey=RYYVLZVUVIJVGH-UHFFFAOYSA-N]
        
        Definition of the graph: There are eleven types of nodes in the graph: Anatomy, Biological Process, Cellular Component, Compound, Disease, Gene, Molecular Function, Pathway, Pharmacologic Class, Side Effect, Symptom.
        Each node has name feature.
        There are these types of edges: Anatomy-downregulates-Gene, Anatomy-expresses-Gene, Anatomy-upregulates-Gene, Compound-binds-Gene, Compound-causes-Side Effect, Compound-downregulates-Gene, Compound-palliates-Disease, Compound-resembles-Compound, Compound-treats-Disease, Compound-upregulates-Gene, Disease-associates-Gene, Disease-downregulates-Gene, Disease-localizes-Anatomy, Disease-presents-Symptom, Disease-resembles-Disease, Disease-upregulates-Gene, Gene-covaries-Gene, Gene-interacts-Gene, Gene-participates-Biological Process, Gene-participates-Cellular Component, Gene-participates-Molecular Function, Gene-participates-Pathway, Gene-regulates-Gene, Pharmacologic Class-includes-Compound.
        Question: How many side effects does Caffeine have?
        Plan 1: The question is asking the number of side effects of a compound node (Caffeine). We need to find the node in the graph and then get the number of the node's Compound-causes-Side Effect.
        Thought 1: The question is related to a compound node (Caffeine). We need to find the node in the graph and calculate the number of the node's 'Compound-causes-Side Effect' neighbors from the graph.
        Action 1: Degree[Retrieve[Caffeine], 'Compound-causes-Side Effect']
        Observation 1: The ID of this node is DB00201. 58
        Plan 2: The question is asking the number of side effects of a compound node (Caffeine). We We have got the answer is 58.
        Thought 2: TThe number of 'Compound-causes-Side Effect' neighbors are 58.
        Action 2: Finish[58]

        Definition of the graph: There are eleven types of nodes in the graph: Anatomy, Biological Process, Cellular Component, Compound, Disease, Gene, Molecular Function, Pathway, Pharmacologic Class, Side Effect, Symptom.
        Each node has name feature.
        There are these types of edges: Anatomy-downregulates-Gene, Anatomy-expresses-Gene, Anatomy-upregulates-Gene, Compound-binds-Gene, Compound-causes-Side Effect, Compound-downregulates-Gene, Compound-palliates-Disease, Compound-resembles-Compound, Compound-treats-Disease, Compound-upregulates-Gene, Disease-associates-Gene, Disease-downregulates-Gene, Disease-localizes-Anatomy, Disease-presents-Symptom, Disease-resembles-Disease, Disease-upregulates-Gene, Gene-covaries-Gene, Gene-interacts-Gene, Gene-participates-Biological Process, Gene-participates-Cellular Component, Gene-participates-Molecular Function, Gene-participates-Pathway, Gene-regulates-Gene, Pharmacologic Class-includes-Compound.
        Question: What compounds can be used to treat Crohn's disease? Please answer the compound names rather than IDs.
        Plan 1: The question is asking the compounds which can be used to treat a disease (Crohn's disease). We need to find the node in the graph and then find the compounds which can be used to treat this disease.
        Thought 1: We need to find the node in the graph.
        Action 1: Retrieve[Crohn's disease]
        Observation 1: The ID of this node is DOID:8778.
        Plan 2: The question is asking the compounds which can be used to treat a disease (Crohn's disease). We have the node of the disease. Thus, we need to find the compounds which can be used to treat this disease.
        Thought 2: We need to check the node's 'Compound-treats-Disease' neighbor from the graph.
        Action 2: Neighbor[DOID:8778, Compound-treats-Disease]
        Observation 2: ['DB01014', 'DB00244', 'DB00795', 'DB00993', 'DB00635', 'DB01033']
        Plan 3: The question is asking the compounds which can be used to treat a disease (Crohn's disease). We have the nodes of the compounds. Thus, we need to get their names.
        Thought 3: The IDs of the compounds are 'DB01014', 'DB00244', 'DB00795', 'DB00993', 'DB00635', 'DB01033'. We need to get their names.
        Action 3: Feature[DB01014, name], Feature[DB00244, name], Feature[DB00795, name], Feature[DB00993, name], Feature[DB00635, name], Feature[DB01033, name]
        Observation 3: Balsalazide, Balsalazide, Mesalazine, Sulfasalazine, Azathioprine, Prednisone, Mercaptopurine
        Plan 4: The question is asking the compounds which can be used to treat a disease (Crohn's disease). We get the answer ———— Balsalazide, Mesalazine, Sulfasalazine, Azathioprine, Prednisone, Mercaptopurine.
        Thought 4: The name of compounds are Balsalazide, Mesalazine, Sulfasalazine, Azathioprine, Prednisone, Mercaptopurine.
        Action 4: Finish[Balsalazide, Mesalazine, Sulfasalazine, Azathioprine, Prednisone, Mercaptopurine]
        
        Definition of the graph: There are eleven types of nodes in the graph: Anatomy, Biological Process, Cellular Component, Compound, Disease, Gene, Molecular Function, Pathway, Pharmacologic Class, Side Effect, Symptom.
        Each node has name feature.
        There are these types of edges: Anatomy-downregulates-Gene, Anatomy-expresses-Gene, Anatomy-upregulates-Gene, Compound-binds-Gene, Compound-causes-Side Effect, Compound-downregulates-Gene, Compound-palliates-Disease, Compound-resembles-Compound, Compound-treats-Disease, Compound-upregulates-Gene, Disease-associates-Gene, Disease-downregulates-Gene, Disease-localizes-Anatomy, Disease-presents-Symptom, Disease-resembles-Disease, Disease-upregulates-Gene, Gene-covaries-Gene, Gene-interacts-Gene, Gene-participates-Biological Process, Gene-participates-Cellular Component, Gene-participates-Molecular Function, Gene-participates-Pathway, Gene-regulates-Gene, Pharmacologic Class-includes-Compound.
        Question: What is the inchikey of Caffeine?
        Plan 1: The question is asking the inchikey of a compound node (Caffeine). We need to find the node in the graph and then get its inchikey.
        Thought 1: We need to find the node in the graph.
        Action 1: Retrieve[Caffeine]
        Observation 1: The ID of this node is DB00201.
        Plan 2: The question is asking the inchikey of a compound node (Caffeine). We have the node. Thus, we need to get its inchikey.
        Thought 2: We need to check the node's 'inchikey' feature from the graph.
        Action 2: Feature[DB00201, inchikey]
        Observation 2: InChIKey=RYYVLZVUVIJVGH-UHFFFAOYSA-N
        Plan 3: The question is asking the inchikey of a compound node (Caffeine). We get the answer ———— InChIKey=RYYVLZVUVIJVGH-UHFFFAOYSA-N.
        Thought 3: The inchikey of the node is InChIKey=RYYVLZVUVIJVGH-UHFFFAOYSA-N.
        Action 3: Finish[InChIKey=RYYVLZVUVIJVGH-UHFFFAOYSA-N]
        
        Definition of the graph: There are eleven types of nodes in the graph: Anatomy, Biological Process, Cellular Component, Compound, Disease, Gene, Molecular Function, Pathway, Pharmacologic Class, Side Effect, Symptom.
        Each node has name feature.
        There are these types of edges: Anatomy-downregulates-Gene, Anatomy-expresses-Gene, Anatomy-upregulates-Gene, Compound-binds-Gene, Compound-causes-Side Effect, Compound-downregulates-Gene, Compound-palliates-Disease, Compound-resembles-Compound, Compound-treats-Disease, Compound-upregulates-Gene, Disease-associates-Gene, Disease-downregulates-Gene, Disease-localizes-Anatomy, Disease-presents-Symptom, Disease-resembles-Disease, Disease-upregulates-Gene, Gene-covaries-Gene, Gene-interacts-Gene, Gene-participates-Biological Process, Gene-participates-Cellular Component, Gene-participates-Molecular Function, Gene-participates-Pathway, Gene-regulates-Gene, Pharmacologic Class-includes-Compound.
        Question: How many side effects does Caffeine have?
        Plan 1: The question is asking the number of side effects of a compound node (Caffeine). We need to find the node in the graph and then get the number of the node's Compound-causes-Side Effect.
        Thought 1: We need to find the node in the graph.
        Action 1: Retrieve[Caffeine]
        Observation 1: The ID of this node is DB00201.
        Plan 2: The question is asking the number of side effects of a compound node (Caffeine). We get the node. Thus, we need to get the number of the node's Compound-causes-Side Effect.
        Thought 2: We need to calculate the number of the node's 'Compound-causes-Side Effect' neighbors from the graph.
        Action 2: Degree[DB00201, 'Compound-causes-Side Effect']
        Observation 2: 58
        Plan 3: The question is asking the number of side effects of a compound node (Caffeine). We get the answer ———— 58.
        Thought 3: The number of 'Compound-causes-Side Effect' neighbors are 58.
        Action 3: Finish[58]
        """,
        "legal": """Definition of the graph: There are four types of nodes in the graph: opinion, opinion_cluster, docket, and court.
        Opinion nodes have features: plain_text. Opinion_cluster nodes have features: syllabus, judges, case_name, attorneys. Docket nodes have features: pacer_case_id, case_name. Court nodes have features: full_name, start_date, end_date, citation_string.
        Opinion nodes are linked to their reference nodes and cited_by nodes, as well as their opinion_cluster nodes. Opinion_cluster nodes are linked to opinion nodes and docket nodes. Docket nodes are linked to opinion_cluster nodes and court nodes. Court nodes are linked to docket nodes.
        Question: Who are the judges that involved in the opinion_cluster: <p>Constitutional law — Title of act — Mortgage—Deed—Liability of grantee — Act of June IB, 1878, P. L. BOB.</p> <p>1. The Act of June 12, 1878, P. L. 205, entitled “An Act to define the liability of grantees of real estate for the incumbrances thereon,” is sufficiently comprehensive in its title to cover both the first and second sections of the act, and does not violate art. Ill, sec. 3 of the constitution of Pennsylvania.</p> <p>2. Where an owner of land creates two mortgages thereon and then conveys the property subject to the mortgages, the grantee covenanting in the deed itself to assume and pay the mortgages as a part of the consideration, and thereafter the grantee conveys the land to another person without expressly assuming any continuing liability to pay the mortgages, such grantee is not liable to the mortgag^ for payment of the mortgages. £V</p>?
        Plan 1: The question is asking the judges of this opinion_cluster node (<p>Constitutional law — Title of act — Mortgage—Deed—Liability of grantee — Act of June IB, 1878, P. L. BOB.</p> <p>1. The Act of June 12, 1878, P. L. 205, entitled “An Act to define the liability of grantees of real estate for the incumbrances thereon,” is sufficiently comprehensive in its title to cover both the first and second sections of the act, and does not violate art. Ill, sec. 3 of the constitution of Pennsylvania.</p> <p>2. Where an owner of land creates two mortgages thereon and then conveys the property subject to the mortgages, the grantee covenanting in the deed itself to assume and pay the mortgages as a part of the consideration, and thereafter the grantee conveys the land to another person without expressly assuming any continuing liability to pay the mortgages, such grantee is not liable to the mortgag^ for payment of the mortgages. £V</p>). We need to find the node in the graph and then get the judges.
        Thought 1: We need to find the node in the graph and check the node's 'judges' feature from the graph.
        Action 1: Feature[Retrieve[<p>Constitutional law — Title of act — Mortgage—Deed—Liability of grantee — Act of June IB, 1878, P. L. BOB.</p> <p>1. The Act of June 12, 1878, P. L. 205, entitled “An Act to define the liability of grantees of real estate for the incumbrances thereon,” is sufficiently comprehensive in its title to cover both the first and second sections of the act, and does not violate art. Ill, sec. 3 of the constitution of Pennsylvania.</p> <p>2. Where an owner of land creates two mortgages thereon and then conveys the property subject to the mortgages, the grantee covenanting in the deed itself to assume and pay the mortgages as a part of the consideration, and thereafter the grantee conveys the land to another person without expressly assuming any continuing liability to pay the mortgages, such grantee is not liable to the mortgag^ for payment of the mortgages. £V</p>], judges]
        Observation 1: The ID of this node is opc-6381448.  Brown, Elkin, Fell, Mestrezat, Moschzisker, Potter, Stewart
        Plan 2: The question is asking the judges of this opinion_cluster node (<p>Constitutional law — Title of act — Mortgage—Deed—Liability of grantee — Act of June IB, 1878, P. L. BOB.</p> <p>1. The Act of June 12, 1878, P. L. 205, entitled “An Act to define the liability of grantees of real estate for the incumbrances thereon,” is sufficiently comprehensive in its title to cover both the first and second sections of the act, and does not violate art. Ill, sec. 3 of the constitution of Pennsylvania.</p> <p>2. Where an owner of land creates two mortgages thereon and then conveys the property subject to the mortgages, the grantee covenanting in the deed itself to assume and pay the mortgages as a part of the consideration, and thereafter the grantee conveys the land to another person without expressly assuming any continuing liability to pay the mortgages, such grantee is not liable to the mortgag^ for payment of the mortgages. £V</p>). We We have got the answer is Brown, Elkin, Fell, Mestrezat, Moschzisker, Potter, Stewart.
        Thought 2: The judges of this opinion cluster is Brown, Elkin, Fell, Mestrezat, Moschzisker, Potter, Stewart
        Action 2: Finish[Brown, Elkin, Fell, Mestrezat, Moschzisker, Potter, Stewart]

        Definition of the graph: There are four types of nodes in the graph: opinion, opinion_cluster, docket, and court.
        Opinion nodes have features: plain_text. Opinion_cluster nodes have features: syllabus, judges, case_name, attorneys. Docket nodes have features: pacer_case_id, case_name. Court nodes have features: full_name, start_date, end_date, citation_string.
        Opinion nodes are linked to their reference nodes and cited_by nodes, as well as their opinion_cluster nodes. Opinion_cluster nodes are linked to opinion nodes and docket nodes. Docket nodes are linked to opinion_cluster nodes and court nodes. Court nodes are linked to docket nodes.
        Question: How many opinions are contained in this opinion_cluster: <p>A general statement that the decision of a state court is against the constitutional rights of the objecting party, or against the Fourteenth Amendment, or that it is without due process of law,-particularly when these objections appear only in specifications of error, so called, will not raise a Federal question, even where the judgment is a final one within Rev. Stat. § 709.</p> <p>In these cases there was no final judgment, such as is provided for in Rev. Stat. § 709, and there does not appear to have arisen any Federal question whatever.</p>'},
        plan 1: The question is asking the number of the opinion neighbors of this node (<p>A general statement that the decision of a state court is against the constitutional rights of the objecting party, or against the Fourteenth Amendment, or that it is without due process of law,-particularly when these objections appear only in specifications of error, so called, will not raise a Federal question, even where the judgment is a final one within Rev. Stat. § 709.</p> <p>In these cases there was no final judgment, such as is provided for in Rev. Stat. § 709, and there does not appear to have arisen any Federal question whatever.</p>). We need to find the node in the graph and count its opinion neighbors.
        Thought 1: We need to find the node in the graph and calculate the node's "opinion" neighbor degree from the graph.
        Action 1: Degree[Retrieve[<p>A general statement that the decision of a state court is against the constitutional rights of the objecting party, or against the Fourteenth Amendment, or that it is without due process of law,-particularly when these objections appear only in specifications of error, so called, will not raise a Federal question, even where the judgment is a final one within Rev. Stat. § 709.</p> <p>In these cases there was no final judgment, such as is provided for in Rev. Stat. § 709, and there does not appear to have arisen any Federal question whatever.</p>], opinion]
        Observation 1: The ID of this node is opc-94592. 1
        Plan 2: The question is asking the number of the opinion neighbors of this node (<p>A general statement that the decision of a state court is against the constitutional rights of the objecting party, or against the Fourteenth Amendment, or that it is without due process of law,-particularly when these objections appear only in specifications of error, so called, will not raise a Federal question, even where the judgment is a final one within Rev. Stat. § 709.</p> <p>In these cases there was no final judgment, such as is provided for in Rev. Stat. § 709, and there does not appear to have arisen any Federal question whatever.</p>). We have got the answer is 1.
        Thought 2: The number of the opinion neighbors is 1
        Action 2: Finish[1]

        Definition of the graph: There are four types of nodes in the graph: opinion, opinion_cluster, docket, and court.
        Opinion nodes have features: plain_text. Opinion_cluster nodes have features: syllabus, judges, case_name, attorneys. Docket nodes have features: pacer_case_id, case_name. Court nodes have features: full_name, start_date, end_date, citation_string.
        Opinion nodes are linked to their reference nodes and cited_by nodes, as well as their opinion_cluster nodes. Opinion_cluster nodes are linked to opinion nodes and docket nodes. Docket nodes are linked to opinion_cluster nodes and court nodes. Court nodes are linked to docket nodes.
        Question: What is the docket pacer case ID for this opinion cluster: In May, 1873, a brewer pays to the collector $100 as special tax, for which a special-tax stamp is issued to Mm. Subsequently he shoivs to. the satisfaction of the Commissions)\' of Internal Revenue that he was not a brewer of more than 500 barrels, and henee that he is entitled to have $50 refunded. The Commissioner issues his certificate for a special-tax stamp of the value of%50 returned. The Treasury refuses payment of the certificate. The breioe)\' brings his action to recover the amount thereof.</p> <p>I. The Commissioner of Internal Revenue being authorized by one section of the statutes (Rev. Stat., § 3220) to refund taxes “unjustly assessed or excessive in, amount," and by another, (§ 3426,) the value of special-tax stamps ‘‘where the rates of duties represented thereby have been paid in error or remitted," it is immaterial whether his certificate be in form under the one section or the other so long as it represents a case where the Commissioner is authorized to refund and the amoirnt which the party is entitled to have refunded.</p> <p>II. When a controversy under the revenue laws is by law required to be determined by other officers and other tribunals, this court is without jurisdiction; but where the proper officer has determined a question, and awarded an allowance, and filed a certificate, and exhausted his jurisdiction, and the Treasury has refused to carry out the award by mating payment, this court has jurisdiction of an action brought to recover the. award..
        Plan 1: The question is asking the judges of this opinion cluster node (In May, 1873, a brewer pays to the collector $100 as special tax, for which a special-tax stamp is issued to Mm. Subsequently he shoivs to. the satisfaction of the Commissions)\' of Internal Revenue that he was not a brewer of more than 500 barrels, and henee that he is entitled to have $50 refunded. The Commissioner issues his certificate for a special-tax stamp of the value of%50 returned. The Treasury refuses payment of the certificate. The breioe)\' brings his action to recover the amount thereof.</p> <p>I. The Commissioner of Internal Revenue being authorized by one section of the statutes (Rev. Stat., § 3220) to refund taxes “unjustly assessed or excessive in, amount," and by another, (§ 3426,) the value of special-tax stamps ‘‘where the rates of duties represented thereby have been paid in error or remitted," it is immaterial whether his certificate be in form under the one section or the other so long as it represents a case where the Commissioner is authorized to refund and the amoirnt which the party is entitled to have refunded.</p> <p>II. When a controversy under the revenue laws is by law required to be determined by other officers and other tribunals, this court is without jurisdiction; but where the proper officer has determined a question, and awarded an allowance, and filed a certificate, and exhausted his jurisdiction, and the Treasury has refused to carry out the award by mating payment, this court has jurisdiction of an action brought to recover the. award.). We need to find the node in the graph and get its judges.
        Thought 1: We need to find the node in the graph and check the node's 'judges' feature from the graph.
        Action 1: Feature[Retrieve[In May, 1873, a brewer pays to the collector $100 as special tax, for which a special-tax stamp is issued to Mm. Subsequently he shoivs to. the satisfaction of the Commissions)\' of Internal Revenue that he was not a brewer of more than 500 barrels, and henee that he is entitled to have $50 refunded. The Commissioner issues his certificate for a special-tax stamp of the value of%50 returned. The Treasury refuses payment of the certificate. The breioe)\' brings his action to recover the amount thereof.</p> <p>I. The Commissioner of Internal Revenue being authorized by one section of the statutes (Rev. Stat., § 3220) to refund taxes “unjustly assessed or excessive in, amount," and by another, (§ 3426,) the value of special-tax stamps ‘‘where the rates of duties represented thereby have been paid in error or remitted," it is immaterial whether his certificate be in form under the one section or the other so long as it represents a case where the Commissioner is authorized to refund and the amoirnt which the party is entitled to have refunded.</p> <p>II. When a controversy under the revenue laws is by law required to be determined by other officers and other tribunals, this court is without jurisdiction; but where the proper officer has determined a question, and awarded an allowance, and filed a certificate, and exhausted his jurisdiction, and the Treasury has refused to carry out the award by mating payment, this court has jurisdiction of an action brought to recover the. award.]]
        Observation 1: The ID of this node is opc-8599951. {'judges': 'Eichaedson', 'case_name': 'Kaufman v. United States', 'attorneys': 'Mr. P. P. Pye for the claimant., Mr. Assistant Attorney-General Simons for the defendants.', 'syllabus': '<p>In May, 1873, a brewer pays to the collector $100 as special tax, for which a special-tax stamp is issued to Mm. Subsequently he shoivs to. the satisfaction of the Commissions)\' of Internal Revenue that he was not a brewer of more than 500 barrels, and henee that he is entitled to have $50 refunded. The Commissioner issues his certificate for a special-tax stamp of the value of%50 returned. The Treasury refuses payment of the certificate. The breioe)\' brings his action to recover the amount thereof.</p> <p>I. The Commissioner of Internal Revenue being authorized by one section of the statutes (Rev. Stat., § 3220) to refund taxes “unjustly assessed or excessive in, amount," and by another, (§ 3426,) the value of special-tax stamps ‘‘where the rates of duties represented thereby have been paid in error or remitted," it is immaterial whether his certificate be in form under the one section or the other so long as it represents a case where the Commissioner is authorized to refund and the amoirnt which the party is entitled to have refunded.</p> <p>II. When a controversy under the revenue laws is by law required to be determined by other officers and other tribunals, this court is without jurisdiction; but where the proper officer has determined a question, and awarded an allowance, and filed a certificate, and exhausted his jurisdiction, and the Treasury has refused to carry out the award by mating payment, this court has jurisdiction of an action brought to recover the. award.</p>'}
        Plan 2: The question is asking the judges of this opinion cluster node (In May, 1873, a brewer pays to the collector $100 as special tax, for which a special-tax stamp is issued to Mm. Subsequently he shoivs to. the satisfaction of the Commissions)\' of Internal Revenue that he was not a brewer of more than 500 barrels, and henee that he is entitled to have $50 refunded. The Commissioner issues his certificate for a special-tax stamp of the value of%50 returned. The Treasury refuses payment of the certificate. The breioe)\' brings his action to recover the amount thereof.</p> <p>I. The Commissioner of Internal Revenue being authorized by one section of the statutes (Rev. Stat., § 3220) to refund taxes “unjustly assessed or excessive in, amount," and by another, (§ 3426,) the value of special-tax stamps ‘‘where the rates of duties represented thereby have been paid in error or remitted," it is immaterial whether his certificate be in form under the one section or the other so long as it represents a case where the Commissioner is authorized to refund and the amoirnt which the party is entitled to have refunded.</p> <p>II. When a controversy under the revenue laws is by law required to be determined by other officers and other tribunals, this court is without jurisdiction; but where the proper officer has determined a question, and awarded an allowance, and filed a certificate, and exhausted his jurisdiction, and the Treasury has refused to carry out the award by mating payment, this court has jurisdiction of an action brought to recover the. award.). We have got the answer is Eichaedson.
        Thought 2: The judges of this opinion cluster is Eichaedson.
        Action 2: Finish[Eichaedson]

        Definition of the graph: There are four types of nodes in the graph: opinion, opinion_cluster, docket, and court.
        Opinion nodes have features: plain_text. Opinion_cluster nodes have features: syllabus, judges, case_name, attorneys. Docket nodes have features: pacer_case_id, case_name. Court nodes have features: full_name, start_date, end_date, citation_string.
        Opinion nodes are linked to their reference nodes and cited_by nodes, as well as their opinion_cluster nodes. Opinion_cluster nodes are linked to opinion nodes and docket nodes. Docket nodes are linked to opinion_cluster nodes and court nodes. Court nodes are linked to docket nodes.
        Question: Who are the judges that involved in the opinion_cluster: <p>Constitutional law — Title of act — Mortgage—Deed—Liability of grantee — Act of June IB, 1878, P. L. BOB.</p> <p>1. The Act of June 12, 1878, P. L. 205, entitled “An Act to define the liability of grantees of real estate for the incumbrances thereon,” is sufficiently comprehensive in its title to cover both the first and second sections of the act, and does not violate art. Ill, sec. 3 of the constitution of Pennsylvania.</p> <p>2. Where an owner of land creates two mortgages thereon and then conveys the property subject to the mortgages, the grantee covenanting in the deed itself to assume and pay the mortgages as a part of the consideration, and thereafter the grantee conveys the land to another person without expressly assuming any continuing liability to pay the mortgages, such grantee is not liable to the mortgag^ for payment of the mortgages. £V</p>?
        Plan 1: The question is asking the judges of this opinion_cluster node (<p>Constitutional law — Title of act — Mortgage—Deed—Liability of grantee — Act of June IB, 1878, P. L. BOB.</p> <p>1. The Act of June 12, 1878, P. L. 205, entitled “An Act to define the liability of grantees of real estate for the incumbrances thereon,” is sufficiently comprehensive in its title to cover both the first and second sections of the act, and does not violate art. Ill, sec. 3 of the constitution of Pennsylvania.</p> <p>2. Where an owner of land creates two mortgages thereon and then conveys the property subject to the mortgages, the grantee covenanting in the deed itself to assume and pay the mortgages as a part of the consideration, and thereafter the grantee conveys the land to another person without expressly assuming any continuing liability to pay the mortgages, such grantee is not liable to the mortgag^ for payment of the mortgages. £V</p>). We need to find the node in the graph and then get the judges.
        Thought 1: We need to find the node in the graph.
        Action 1: Retrieve[<p>Constitutional law — Title of act — Mortgage—Deed—Liability of grantee — Act of June IB, 1878, P. L. BOB.</p> <p>1. The Act of June 12, 1878, P. L. 205, entitled “An Act to define the liability of grantees of real estate for the incumbrances thereon,” is sufficiently comprehensive in its title to cover both the first and second sections of the act, and does not violate art. Ill, sec. 3 of the constitution of Pennsylvania.</p> <p>2. Where an owner of land creates two mortgages thereon and then conveys the property subject to the mortgages, the grantee covenanting in the deed itself to assume and pay the mortgages as a part of the consideration, and thereafter the grantee conveys the land to another person without expressly assuming any continuing liability to pay the mortgages, such grantee is not liable to the mortgag^ for payment of the mortgages. £V</p>]
        Observation 1: The ID of this node is opc-6381448.
        Plan 2: The question is asking the judges of this opinion_cluster node (<p>Constitutional law — Title of act — Mortgage—Deed—Liability of grantee — Act of June IB, 1878, P. L. BOB.</p> <p>1. The Act of June 12, 1878, P. L. 205, entitled “An Act to define the liability of grantees of real estate for the incumbrances thereon,” is sufficiently comprehensive in its title to cover both the first and second sections of the act, and does not violate art. Ill, sec. 3 of the constitution of Pennsylvania.</p> <p>2. Where an owner of land creates two mortgages thereon and then conveys the property subject to the mortgages, the grantee covenanting in the deed itself to assume and pay the mortgages as a part of the consideration, and thereafter the grantee conveys the land to another person without expressly assuming any continuing liability to pay the mortgages, such grantee is not liable to the mortgag^ for payment of the mortgages. £V</p>). We have the node. Thus, we need to get the judges.
        Thought 2: We need to check the node's 'judges' feature from the graph.
        Action 2: Feature[opc-6381448, judges]
        Observation 2: Brown, Elkin, Fell, Mestrezat, Moschzisker, Potter, Stewart
        Plan 3: The question is asking the judges of this opinion_cluster node (<p>Constitutional law — Title of act — Mortgage—Deed—Liability of grantee — Act of June IB, 1878, P. L. BOB.</p> <p>1. The Act of June 12, 1878, P. L. 205, entitled “An Act to define the liability of grantees of real estate for the incumbrances thereon,” is sufficiently comprehensive in its title to cover both the first and second sections of the act, and does not violate art. Ill, sec. 3 of the constitution of Pennsylvania.</p> <p>2. Where an owner of land creates two mortgages thereon and then conveys the property subject to the mortgages, the grantee covenanting in the deed itself to assume and pay the mortgages as a part of the consideration, and thereafter the grantee conveys the land to another person without expressly assuming any continuing liability to pay the mortgages, such grantee is not liable to the mortgag^ for payment of the mortgages. £V</p>). We get the answer ———— Brown, Elkin, Fell, Mestrezat, Moschzisker, Potter, Stewart.
        Thought 3: The judges of this opinion cluster is Brown, Elkin, Fell, Mestrezat, Moschzisker, Potter, Stewart
        Action 3: Finish[Brown, Elkin, Fell, Mestrezat, Moschzisker, Potter, Stewart]

        Definition of the graph: There are four types of nodes in the graph: opinion, opinion_cluster, docket, and court.
        Opinion nodes have features: plain_text. Opinion_cluster nodes have features: syllabus, judges, case_name, attorneys. Docket nodes have features: pacer_case_id, case_name. Court nodes have features: full_name, start_date, end_date, citation_string.
        Opinion nodes are linked to their reference nodes and cited_by nodes, as well as their opinion_cluster nodes. Opinion_cluster nodes are linked to opinion nodes and docket nodes. Docket nodes are linked to opinion_cluster nodes and court nodes. Court nodes are linked to docket nodes.
        Question: How many opinions are contained in this opinion_cluster: <p>A general statement that the decision of a state court is against the constitutional rights of the objecting party, or against the Fourteenth Amendment, or that it is without due process of law,-particularly when these objections appear only in specifications of error, so called, will not raise a Federal question, even where the judgment is a final one within Rev. Stat. § 709.</p> <p>In these cases there was no final judgment, such as is provided for in Rev. Stat. § 709, and there does not appear to have arisen any Federal question whatever.</p>'},
        Plan 1: The question is asking the number of the opinion neighbors of this node (<p>A general statement that the decision of a state court is against the constitutional rights of the objecting party, or against the Fourteenth Amendment, or that it is without due process of law,-particularly when these objections appear only in specifications of error, so called, will not raise a Federal question, even where the judgment is a final one within Rev. Stat. § 709.</p> <p>In these cases there was no final judgment, such as is provided for in Rev. Stat. § 709, and there does not appear to have arisen any Federal question whatever.</p>). We need to find the node in the graph and count its opinion neighbors.
        Thought 1: We need to find the node in the graph.
        Action 1: Retrieve[<p>A general statement that the decision of a state court is against the constitutional rights of the objecting party, or against the Fourteenth Amendment, or that it is without due process of law,-particularly when these objections appear only in specifications of error, so called, will not raise a Federal question, even where the judgment is a final one within Rev. Stat. § 709.</p> <p>In these cases there was no final judgment, such as is provided for in Rev. Stat. § 709, and there does not appear to have arisen any Federal question whatever.</p>]
        Observation 1: The ID of this node is opc-94592.
        Plan 2: The question is asking the number of the opinion neighbors of this node (<p>A general statement that the decision of a state court is against the constitutional rights of the objecting party, or against the Fourteenth Amendment, or that it is without due process of law,-particularly when these objections appear only in specifications of error, so called, will not raise a Federal question, even where the judgment is a final one within Rev. Stat. § 709.</p> <p>In these cases there was no final judgment, such as is provided for in Rev. Stat. § 709, and there does not appear to have arisen any Federal question whatever.</p>). We have the node. Thus, we need to count its opinion neighbors.
        Thought 2: We need to calculate the node's "opinion" neighbor degree from the graph.
        Action 2: Degree[opc-94592, opinion]
        Observation 2: 1
        Plan 3: The question is asking the number of the opinion neighbors of this node (<p>A general statement that the decision of a state court is against the constitutional rights of the objecting party, or against the Fourteenth Amendment, or that it is without due process of law,-particularly when these objections appear only in specifications of error, so called, will not raise a Federal question, even where the judgment is a final one within Rev. Stat. § 709.</p> <p>In these cases there was no final judgment, such as is provided for in Rev. Stat. § 709, and there does not appear to have arisen any Federal question whatever.</p>). We get the answer ———— 1
        Thought 3: The number of the opinion neighbors is 1
        Action 3: Finish[1]

        Definition of the graph: There are four types of nodes in the graph: opinion, opinion_cluster, docket, and court.
        Opinion nodes have features: plain_text. Opinion_cluster nodes have features: syllabus, judges, case_name, attorneys. Docket nodes have features: pacer_case_id, case_name. Court nodes have features: full_name, start_date, end_date, citation_string.
        Opinion nodes are linked to their reference nodes and cited_by nodes, as well as their opinion_cluster nodes. Opinion_cluster nodes are linked to opinion nodes and docket nodes. Docket nodes are linked to opinion_cluster nodes and court nodes. Court nodes are linked to docket nodes.
        Question: What is the docket pacer case ID for this opinion cluster: In May, 1873, a brewer pays to the collector $100 as special tax, for which a special-tax stamp is issued to Mm. Subsequently he shoivs to. the satisfaction of the Commissions)\' of Internal Revenue that he was not a brewer of more than 500 barrels, and henee that he is entitled to have $50 refunded. The Commissioner issues his certificate for a special-tax stamp of the value of%50 returned. The Treasury refuses payment of the certificate. The breioe)\' brings his action to recover the amount thereof.</p> <p>I. The Commissioner of Internal Revenue being authorized by one section of the statutes (Rev. Stat., § 3220) to refund taxes “unjustly assessed or excessive in, amount," and by another, (§ 3426,) the value of special-tax stamps ‘‘where the rates of duties represented thereby have been paid in error or remitted," it is immaterial whether his certificate be in form under the one section or the other so long as it represents a case where the Commissioner is authorized to refund and the amoirnt which the party is entitled to have refunded.</p> <p>II. When a controversy under the revenue laws is by law required to be determined by other officers and other tribunals, this court is without jurisdiction; but where the proper officer has determined a question, and awarded an allowance, and filed a certificate, and exhausted his jurisdiction, and the Treasury has refused to carry out the award by mating payment, this court has jurisdiction of an action brought to recover the. award..
        Plan 1: The question is asking the judges of this opinion cluster node (In May, 1873, a brewer pays to the collector $100 as special tax, for which a special-tax stamp is issued to Mm. Subsequently he shoivs to. the satisfaction of the Commissions)\' of Internal Revenue that he was not a brewer of more than 500 barrels, and henee that he is entitled to have $50 refunded. The Commissioner issues his certificate for a special-tax stamp of the value of%50 returned. The Treasury refuses payment of the certificate. The breioe)\' brings his action to recover the amount thereof.</p> <p>I. The Commissioner of Internal Revenue being authorized by one section of the statutes (Rev. Stat., § 3220) to refund taxes “unjustly assessed or excessive in, amount," and by another, (§ 3426,) the value of special-tax stamps ‘‘where the rates of duties represented thereby have been paid in error or remitted," it is immaterial whether his certificate be in form under the one section or the other so long as it represents a case where the Commissioner is authorized to refund and the amoirnt which the party is entitled to have refunded.</p> <p>II. When a controversy under the revenue laws is by law required to be determined by other officers and other tribunals, this court is without jurisdiction; but where the proper officer has determined a question, and awarded an allowance, and filed a certificate, and exhausted his jurisdiction, and the Treasury has refused to carry out the award by mating payment, this court has jurisdiction of an action brought to recover the. award.). We need to find the node in the graph and get its judges.
        Thought 1: We need to find the node in the graph.
        Action 1: Retrieve[In May, 1873, a brewer pays to the collector $100 as special tax, for which a special-tax stamp is issued to Mm. Subsequently he shoivs to. the satisfaction of the Commissions)\' of Internal Revenue that he was not a brewer of more than 500 barrels, and henee that he is entitled to have $50 refunded. The Commissioner issues his certificate for a special-tax stamp of the value of%50 returned. The Treasury refuses payment of the certificate. The breioe)\' brings his action to recover the amount thereof.</p> <p>I. The Commissioner of Internal Revenue being authorized by one section of the statutes (Rev. Stat., § 3220) to refund taxes “unjustly assessed or excessive in, amount," and by another, (§ 3426,) the value of special-tax stamps ‘‘where the rates of duties represented thereby have been paid in error or remitted," it is immaterial whether his certificate be in form under the one section or the other so long as it represents a case where the Commissioner is authorized to refund and the amoirnt which the party is entitled to have refunded.</p> <p>II. When a controversy under the revenue laws is by law required to be determined by other officers and other tribunals, this court is without jurisdiction; but where the proper officer has determined a question, and awarded an allowance, and filed a certificate, and exhausted his jurisdiction, and the Treasury has refused to carry out the award by mating payment, this court has jurisdiction of an action brought to recover the. award.]
        Observation 1: The ID of this node is opc-8599951.
        Plan 2: The question is asking the judges of this opinion cluster node (In May, 1873, a brewer pays to the collector $100 as special tax, for which a special-tax stamp is issued to Mm. Subsequently he shoivs to. the satisfaction of the Commissions)\' of Internal Revenue that he was not a brewer of more than 500 barrels, and henee that he is entitled to have $50 refunded. The Commissioner issues his certificate for a special-tax stamp of the value of%50 returned. The Treasury refuses payment of the certificate. The breioe)\' brings his action to recover the amount thereof.</p> <p>I. The Commissioner of Internal Revenue being authorized by one section of the statutes (Rev. Stat., § 3220) to refund taxes “unjustly assessed or excessive in, amount," and by another, (§ 3426,) the value of special-tax stamps ‘‘where the rates of duties represented thereby have been paid in error or remitted," it is immaterial whether his certificate be in form under the one section or the other so long as it represents a case where the Commissioner is authorized to refund and the amoirnt which the party is entitled to have refunded.</p> <p>II. When a controversy under the revenue laws is by law required to be determined by other officers and other tribunals, this court is without jurisdiction; but where the proper officer has determined a question, and awarded an allowance, and filed a certificate, and exhausted his jurisdiction, and the Treasury has refused to carry out the award by mating payment, this court has jurisdiction of an action brought to recover the. award.). We have the node. Thus, we need to get its judges.
        Thought 2: We need to check the node's 'judges' feature from the graph.
        Action 2: Feature[opc-8599951]
        Observation 2: {'judges': 'Eichaedson', 'case_name': 'Kaufman v. United States', 'attorneys': 'Mr. P. P. Pye for the claimant., Mr. Assistant Attorney-General Simons for the defendants.', 'syllabus': '<p>In May, 1873, a brewer pays to the collector $100 as special tax, for which a special-tax stamp is issued to Mm. Subsequently he shoivs to. the satisfaction of the Commissions)\' of Internal Revenue that he was not a brewer of more than 500 barrels, and henee that he is entitled to have $50 refunded. The Commissioner issues his certificate for a special-tax stamp of the value of%50 returned. The Treasury refuses payment of the certificate. The breioe)\' brings his action to recover the amount thereof.</p> <p>I. The Commissioner of Internal Revenue being authorized by one section of the statutes (Rev. Stat., § 3220) to refund taxes “unjustly assessed or excessive in, amount," and by another, (§ 3426,) the value of special-tax stamps ‘‘where the rates of duties represented thereby have been paid in error or remitted," it is immaterial whether his certificate be in form under the one section or the other so long as it represents a case where the Commissioner is authorized to refund and the amoirnt which the party is entitled to have refunded.</p> <p>II. When a controversy under the revenue laws is by law required to be determined by other officers and other tribunals, this court is without jurisdiction; but where the proper officer has determined a question, and awarded an allowance, and filed a certificate, and exhausted his jurisdiction, and the Treasury has refused to carry out the award by mating payment, this court has jurisdiction of an action brought to recover the. award.</p>'}
        Plan 3: The question is asking the judges of this opinion cluster node (In May, 1873, a brewer pays to the collector $100 as special tax, for which a special-tax stamp is issued to Mm. Subsequently he shoivs to. the satisfaction of the Commissions)\' of Internal Revenue that he was not a brewer of more than 500 barrels, and henee that he is entitled to have $50 refunded. The Commissioner issues his certificate for a special-tax stamp of the value of%50 returned. The Treasury refuses payment of the certificate. The breioe)\' brings his action to recover the amount thereof.</p> <p>I. The Commissioner of Internal Revenue being authorized by one section of the statutes (Rev. Stat., § 3220) to refund taxes “unjustly assessed or excessive in, amount," and by another, (§ 3426,) the value of special-tax stamps ‘‘where the rates of duties represented thereby have been paid in error or remitted," it is immaterial whether his certificate be in form under the one section or the other so long as it represents a case where the Commissioner is authorized to refund and the amoirnt which the party is entitled to have refunded.</p> <p>II. When a controversy under the revenue laws is by law required to be determined by other officers and other tribunals, this court is without jurisdiction; but where the proper officer has determined a question, and awarded an allowance, and filed a certificate, and exhausted his jurisdiction, and the Treasury has refused to carry out the award by mating payment, this court has jurisdiction of an action brought to recover the. award.). We get the answer ———— Eichaedson.
        Thought 3: The judges of this opinion cluster is Eichaedson
        Action 3: Finish[Eichaedson]
        """,
    "goodreads": """Definition of the graph: There are four types of nodes in the graph: book, author, publisher, and series. 
        Book nodes have features: country_code, language_code, is_ebook, title, description, format, num_pages, publication_year, url, popular_shelves, and genres. Author nodes have features: name. Publisher nodes have features: name. Series nodes have features: title and description. 
        Book nodes are linked to their author nodes, publisher nodes, series nodes and similar_books nodes. Author nodes are linked to their book nodes. Publisher nodes are linked to their book nodes. Series nodes are linked to their book nodes.
        Question: Who is the author of the book 'The Great Gatsby'?
        Plan 1: The question is asking for the author of a specific book (The Great Gatsby). We need to find the book node and then get its author.
        Thought 1: We need to find the book node in the graph and check the book node's author neighbor from the graph.
        Action 1: Neighbor[Retrieve[The Great Gatsby], author]
        Observation 1: The ID of this book node is 21366268. ['3190']
        Plan 2: The question is asking for the author of a specific book (The Great Gatsby). We have the author's ID. Thus, we need to get the author's name.
        Thought 2: The ID of the author is 3190. We need to check the name.
        Action 2: Feature[3190, name]
        Observation 2: F. Scott Fitzgerald
        Plan 3: The question is asking for the author of a specific book (The Great Gatsby). We have got the answer is F. Scott Fitzgerald.
        Thought 3: The author of 'The Great Gatsby' is F. Scott Fitzgerald.
        Action 3: Finish[F. Scott Fitzgerald]
        
        Definition of the graph: There are four types of nodes in the graph: book, author, publisher, and series. 
        Book nodes have features: country_code, language_code, is_ebook, title, description, format, num_pages, publication_year, url, popular_shelves, and genres. Author nodes have features: name. Publisher nodes have features: name. Series nodes have features: title and description. 
        Book nodes are linked to their author nodes, publisher nodes, series nodes and similar_books nodes. Author nodes are linked to their book nodes. Publisher nodes are linked to their book nodes. Series nodes are linked to their book nodes.
        Question: What is the genre of the book '1984'?
        Plan 1: The question is asking for the genre of a specific book (1984). We need to find the book node and then get its genre.
        Thought 1: We need to find the book node in the graph and then check the book node's genre feature from the graph.
        Action 1: Feature[Retrieve[1984], genres]
        Observation 1: The ID of this book node is 3475269. currently-reading, classics, fiction
        Plan 2: The question is asking for the genre of a specific book (1984). We have got the answer is currently-reading, classics, fiction.
        Thought 2: The genres of the book '1984' are currently-reading, classics, fiction.
        Action 2: Finish[currently-reading, classics, fiction]
        
        Definition of the graph: There are four types of nodes in the graph: book, author, publisher, and series. 
        Book nodes have features: country_code, language_code, is_ebook, title, description, format, num_pages, publication_year, url, popular_shelves, and genres. Author nodes have features: name. Publisher nodes have features: name. Series nodes have features: title and description. 
        Book nodes are linked to their author nodes, publisher nodes, series nodes and similar_books nodes. Author nodes are linked to their book nodes. Publisher nodes are linked to their book nodes. Series nodes are linked to their book nodes.
        Question: How many series included the book 'Harry Potter and the Sorcerer's Stone'?
        Plan 1: The question is asking for the number of series included in a specific book (Harry Potter and the Sorcerer's Stone). We need to find the book node and then get its series neighbor degree.
        Thought 1: We need to find the book node for 'Harry Potter and the Sorcerer's Stone' to determine its series and calculate the node's series neighbor degree from the graph.
        Action 1: Degree[Retrieve[Harry Potter and the Sorcerer's Stone], series]
        Observation 1: The ID of this book node is 27421523. 1
        Plan 2: The question is asking for the number of series included in a specific book (Harry Potter and the Sorcerer's Stone). We have got the answer is 1.
        Thought 2: The number of series neighbors is 1.
        Action 2: Finish[1]

        Definition of the graph: There are four types of nodes in the graph: book, author, publisher, and series. 
        Book nodes have features: country_code, language_code, is_ebook, title, description, format, num_pages, publication_year, url, popular_shelves, and genres. Author nodes have features: name. Publisher nodes have features: name. Series nodes have features: title and description. 
        Book nodes are linked to their author nodes, publisher nodes, series nodes and similar_books nodes. Author nodes are linked to their book nodes. Publisher nodes are linked to their book nodes. Series nodes are linked to their book nodes.
        Question: Who is the author of the book 'The Great Gatsby'?
        Plan 1: The question is asking for the author of a specific book (The Great Gatsby). We need to find the book node and then get its author.
        Thought 1: We need to find the book node in the graph.
        Action 1: Retrieve[The Great Gatsby]
        Observation 1: The ID of this book node is 21366268.
        Plan 2: The question is asking for the author of a specific book (The Great Gatsby). We have the book node. Thus, we need to get its author.
        Thought 2: We need to check the book node's author neighbor from the graph.
        Action 2: Neighbor[21366268, author]
        Observation 2: ['3190']
        Plan 3: The question is asking for the author of a specific book (The Great Gatsby). We have the author node. Thus, we need to get its name.
        Thought 3: The ID of the author is 3190. We need to check the name.
        Action 3: Feature[3190, name]
        Observation 3: F. Scott Fitzgerald
        Plan 4: The question is asking for the author of a specific book (The Great Gatsby). We get the answer ———— F. Scott Fitzgerald
        Thought 4: The author of 'The Great Gatsby' is F. Scott Fitzgerald.
        Action 4: Finish[F. Scott Fitzgerald]
        
        Definition of the graph: There are four types of nodes in the graph: book, author, publisher, and series. 
        Book nodes have features: country_code, language_code, is_ebook, title, description, format, num_pages, publication_year, url, popular_shelves, and genres. Author nodes have features: name. Publisher nodes have features: name. Series nodes have features: title and description. 
        Book nodes are linked to their author nodes, publisher nodes, series nodes and similar_books nodes. Author nodes are linked to their book nodes. Publisher nodes are linked to their book nodes. Series nodes are linked to their book nodes.
        Question: What is the genre of the book '1984'?
        Plan 1: The question is asking for the genre of a specific book (1984). We need to find the book node and then get its genre.
        Thought 1: We need to find the book node in the graph.
        Action 1: Retrieve[1984]
        Observation 1: The ID of this book node is 3475269.
        Plan 2: The question is asking for the genre of a specific book (1984). We have the book node. Thus, we need to get its genre.
        Thought 2: We need to check the book node's genre feature from the graph.
        Action 2: Feature[3475269, genres]
        Observation 2: currently-reading, classics, fiction
        Plan 3: The question is asking for the genre of a specific book (1984). We have the answer ———— currently-reading, classics, fiction.
        Thought 3: The genres of the book '1984' are currently-reading, classics, fiction.
        Action 3: Finish[currently-reading, classics, fiction]
        
        Definition of the graph: There are four types of nodes in the graph: book, author, publisher, and series. 
        Book nodes have features: country_code, language_code, is_ebook, title, description, format, num_pages, publication_year, url, popular_shelves, and genres. Author nodes have features: name. Publisher nodes have features: name. Series nodes have features: title and description. 
        Book nodes are linked to their author nodes, publisher nodes, series nodes and similar_books nodes. Author nodes are linked to their book nodes. Publisher nodes are linked to their book nodes. Series nodes are linked to their book nodes.
        Question: How many series included the book 'Harry Potter and the Sorcerer's Stone'?
        Plan 1: The question is asking for the number of series included in a specific book (Harry Potter and the Sorcerer's Stone). We need to find the book node and then get its series neighbor degree.
        Thought 1: We need to find the book node for 'Harry Potter and the Sorcerer's Stone' to determine its series.
        Action 1: Retrieve[Harry Potter and the Sorcerer's Stone]
        Observation 1: The ID of this book node is 27421523.
        Plan 2: The question is asking for the number of series included in a specific book (Harry Potter and the Sorcerer's Stone). We have the book node. Thus, we need to get its series neighbor degree.
        Thought 2: We need to calculate the node's series neighbor degree from the graph.
        Action 2: Degree[27421523, series]
        Observation 2: 1
        Plan 3: The question is asking for the number of series included in a specific book (Harry Potter and the Sorcerer's Stone). We have the answer ———— 1.
        Thought 3: The number of series neighbors is 1.
        Action 3: Finish[1]
        """,
    "dblp": """Definition of the graph: There are three types of nodes in the graph: paper, author and venue.
        Paper nodes have features: title, abstract, keywords, lang, and year. Author nodes have features: name and organization. Venue nodes have features: name.
        Paper nodes are linked to their author nodes, venue nodes, reference nodes (the papers this paper cite) and cited_by nodes (other papers which cite this paper). Author nodes are linked to their paper nodes. Venue nodes are linked to their paper nodes.
        Question: What organization is researcher Greg Daville from?
        Plan 1: The question is asking about the organization of a researcher named Greg Daville. We need to find the node of Greg Daville in the graph and check the node's organization feature from the graph.
        Thought 1: We need to find the node corresponding to Greg Daville in the graph and check the node's organization feature from the graph.
        Action 1: Feature[Retrieve[Greg Daville], organization]
        Observation 1: The ID of this retrieval target node is 53f460a7dabfaee4dc83702a. Hove East Sussex, United Kingdom
        Plan 2: The question is asking about the organization of a researcher named Greg Daville. We have got the answer is Hove East Sussex, United Kingdom.
        Thought 2: The organization of researcher Greg Daville is Hove East Sussex, United Kingdom.
        Action 2: Finish[Hove East Sussex, United Kingdom]
    
        Definition of the graph: There are three types of nodes in the graph: paper, author and venue.
        Paper nodes have features: title, abstract, keywords, lang, and year. Author nodes have features: name and organization. Venue nodes have features: name.
        Paper nodes are linked to their author nodes, venue nodes, reference nodes (the papers this paper cite) and cited_by nodes (other papers which cite this paper). Author nodes are linked to their paper nodes. Venue nodes are linked to their paper nodes.
        Question: How many papers are written by author Nicholas Lydon?
        Plan 1: The question is asking for the number of papers written by a specific author (Nicholas Lydon). We need to find the author node in the graph and then calculate the number of the author's papers.
        Thought 1: We need to find the author node in the graph and calculate the "paper" neighbor degree of this node.
        Action 1: Degree[Retrieve[Nicholas Lydon], paper]
        Observation 1: The ID of this retrieval target node is 53f438c3dabfaedf43596117. 2
        Plan 2: The question is asking for the number of papers written by a specific author (Nicholas Lydon). We have got the answer is 2.
        Thought 2: The number of the paper neighbors is 2
        Action 2: Finish[2]

        Definition of the graph: There are three types of nodes in the graph: paper, author and venue.
        Paper nodes have features: title, abstract, keywords, lang, and year. Author nodes have features: name and organization. Venue nodes have features: name.
        Paper nodes are linked to their author nodes, venue nodes, reference nodes (the papers this paper cite) and cited_by nodes (other papers which cite this paper). Author nodes are linked to their paper nodes. Venue nodes are linked to their paper nodes.
        Question: What was the publish venue of the paper Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars?
        Plan 1: The question is asking the published venue of a paper (Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars). We need to find the paper node and then get its venue.
        Thought 1: We need to find the node in the graph.
        Action 1: Retrieve[Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars]
        Observation 1: The ID of this node is 2090642949.
        Plan 2: The question is asking the published venue of a paper (Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars). We have the ID of the paper node. Thus, we need to get the venue's name.
        Thought 2: We need to check the node's venue neighbor from the graph and get their names.
        Action 2: Feature[Neighbor[2090642949, venue], name]
        Observation 2: ['1980519', '1053242'] the astrophysical journal, the atmosphere journal
        Plan 3: The question is asking the published venue of a paper (Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars). We have got the answer is the astrophysical journal, the atmosphere journal
        Thought 3: The name of the published venues are the astrophysical journal and the atmosphere journal
        Action 3: Finish[the astrophysical journal, the atmosphere journal]

        Definition of the graph: There are three types of nodes in the graph: paper, author and venue.
        Paper nodes have features: title, abstract, keywords, lang, and year. Author nodes have features: name and organization. Venue nodes have features: name.
        Paper nodes are linked to their author nodes, venue nodes, reference nodes (the papers this paper cite) and cited_by nodes (other papers which cite this paper). Author nodes are linked to their paper nodes. Venue nodes are linked to their paper nodes.
        Question: What organization is researcher Greg Daville from?
        Plan 1: The question is asking about the organization of a researcher named Greg Daville. We need to find the node of Greg Daville in the graph and check the node's organization feature from the graph.
        Thought 1: We need to find the node corresponding to Greg Daville in the graph.
        Action 1: Retrieve[Greg Daville]
        Observation 1: The ID of this retrieval target node is 53f460a7dabfaee4dc83702a.
        Plan 2: The question is asking about the organization of a researcher named Greg Daville. We have the node of Greg Daville. Thus, we need to check the node's organization feature from the graph.
        Thought 2: We need to check the node's organization feature from the graph.
        Action 2: Feature[53f460a7dabfaee4dc83702a, organization]
        Observation 2: Hove East Sussex, United Kingdom 
        Plan 3: The question is asking about the organization of a researcher named Greg Daville. We have the answer ———— Hove East Sussex, United Kingdom
        Thought 3: The organization of researcher Greg Daville is Hove East Sussex, United Kingdom.
        Action 3: Finish[Hove East Sussex, United Kingdom] 
    
        Definition of the graph: There are three types of nodes in the graph: paper, author and venue.
        Paper nodes have features: title, abstract, keywords, lang, and year. Author nodes have features: name and organization. Venue nodes have features: name.
        Paper nodes are linked to their author nodes, venue nodes, reference nodes (the papers this paper cite) and cited_by nodes (other papers which cite this paper). Author nodes are linked to their paper nodes. Venue nodes are linked to their paper nodes.
        Question: How many papers are written by author Nicholas Lydon?
        Plan 1: The question is asking for the number of papers written by a specific author (Nicholas Lydon). We need to find the author node in the graph and then calculate the number of the author's papers.
        Thought 1: We need to find the author node in the graph.
        Action 1: Retrieve[Nicholas Lydon]
        Observation 1: The ID of this retrieval target node is 53f438c3dabfaedf43596117.
        Plan 2: The question is asking for the number of papers written by a specific author (Nicholas Lydon). We have the anthor node. Thus, we need to calculate the number of the author's papers.
        Thought 2: We need to calculate the "paper" neighbor degree of this node.
        Action 2: Degree[53f438c3dabfaedf43596117, paper]
        Observation 2: 2
        Plan 3: The question is asking for the number of papers written by a specific author (Nicholas Lydon). We have the answer ———— 2
        Thought 3: The number of the paper neighbors is 2
        Action 3: Finish[2]        

        Definition of the graph: There are three types of nodes in the graph: paper, author and venue.
        Paper nodes have features: title, abstract, keywords, lang, and year. Author nodes have features: name and organization. Venue nodes have features: name.
        Paper nodes are linked to their author nodes, venue nodes, reference nodes (the papers this paper cite) and cited_by nodes (other papers which cite this paper). Author nodes are linked to their paper nodes. Venue nodes are linked to their paper nodes.
        Question: What was the publish venue of the paper Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars?
        Plan 1: The question is asking the published venue of a paper (Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars). We need to find the paper node and then get its venue.
        Thought 1: We need to find the node in the graph.
        Action 1: Retrieve[Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars]
        Observation 1: The ID of this node is 2090642949.
        Plan 2: The question is asking the published venue of a paper (Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars). We have the ID of the paper node. Thus, we need to get the venue's name.
        Thought 2: We need to check the node's venue neighbor from the graph.
        Action 2: Neighbor[2090642949, venue]
        Observation 2: ['1980519', '1053242']
        Plan 3: The question is asking the published venue of a paper (Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars). We have the ID of the venue. Thus, we need to get the name of venue.
        Thought 3: The ID of the published venue are 1980519 and 1053242. We need to get their names.
        Action 3: Feature[1980519, name], Feature[1053242, name]
        Observation 3: the astrophysical journal, the atmosphere journal
        Plan 4: The question is asking the published venue of a paper (Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars). We have the answer ———— the astrophysical journal, the atmosphere journal
        Thought 4: The name of the published venues are the astrophysical journal and the atmosphere journal
        Action 4: Finish[the astrophysical journal, the atmosphere journal]
        """,
}

PLAN_ONLY_EXAMPLES = {
    "maple": """Definition of the graph: There are three types of nodes in the graph: paper, author and venue.
        Paper nodes have features: title, abstract, year and label. Author nodes have features: name. Venue nodes have features: name.
        Paper nodes are linked to their author nodes, venue nodes, reference paper nodes and cited_by paper nodes. Author nodes are linked to their paper nodes. Venue nodes are linked to their paper nodes.
        Question: When was the paper Strongly Interacting Higgs Sector in the Minimal Standard Model published?
        Plan 1: The question is asking the published time of a node (Strongly Interacting Higgs Sector in the Minimal Standard Model). Thus, we need to locate the relevant paper and extract the published year from the graph.
        Thought 1: We need to find the node in the graph.
        Action 1: Retrieve[Strongly Interacting Higgs Sector in the Minimal Standard Model]
        Observation 1: The ID of this node is 3101448248.
        Plan 2: The question is asking the published time of a node (Strongly Interacting Higgs Sector in the Minimal Standard Model). We have the paper node. Thus, we need to extract the published year from the graph.
        Thought 2: The question is asking the published date of a paper, we need to check the node feature (year) from the graph.
        Action 2: Feature[3101448248, year]
        Observation 2: 1993
        Plan 3: The question is asking the published time of a node (Strongly Interacting Higgs Sector in the Minimal Standard Model). We have the answer ———— 1993.
        Thought 3: The published date of the paper is 1993.
        Action 3: Finish[1993]

        Definition of the graph: There are three types of nodes in the graph: paper, author and venue.
        Paper nodes have features: title, abstract, year and label. Author nodes have features: name. Venue nodes have features: name.
        Paper nodes are linked to their author nodes, venue nodes, reference paper nodes and cited_by paper nodes. Author nodes are linked to their paper nodes. Venue nodes are linked to their paper nodes.
        Question: How many authors do the paper Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars have?
        Plan 1: The question is asking the number of authors of a paper (Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars). Thus, we need to locate the relevant paper and count the authors from the graph.
        Thought 1: We need to find the node in the graph.
        Action 1: Retrieve[Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars]
        Observation 1: The ID of this node is 2090642949.
        Plan 2: The question is asking the number of authors of a paper (Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars). We have the paper node. Thus, we need to count the authors from the graph.
        Thought 2: We need to calculate the node's author neighbor degree from the graph.
        Action 2: Degree[2090642949, author]
        Observation 2: 2
        Plan 3: The question is asking the number of authors of a paper (Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars). We get the answer ———— 2.
        Thought 3: The number of the authors is 2
        Action 3: Finish[2]

        Definition of the graph: There are three types of nodes in the graph: paper, author and venue.
        Paper nodes have features: title, abstract, year and label. Author nodes have features: name. Venue nodes have features: name.
        Paper nodes are linked to their author nodes, venue nodes, reference paper nodes and cited_by paper nodes. Author nodes are linked to their paper nodes. Venue nodes are linked to their paper nodes.
        Question: What was the publish venue of the paper Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars?
        Plan 1: The question is asking information of a node (Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars). Thus, we need to locate the relevant paper and get the venue name of the paper from the graph.
        Thought 1: We need to find the node in the graph.
        Action 1: Retrieve[Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars]
        Observation 1: The ID of this node is 2090642949.
        Plan 2: The question is asking information of a node (Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars). We have the paper node. Thus, we need to get venue neighbor of the paper.
        Thought 2: We need to check the node's venue neighbor from the graph.
        Action 2: Neighbor[2090642949, venue]
        Observation 2: ['1980519', '1053242']
        Plan 3: The question is asking information of a node (Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars). We have the venues node. Thus, we need to get the names of these venues.
        Thought 3: The ID of the published venue are 1980519 and 1053242. We need to get their names.
        Action 3: Feature[1980519, name], Feature[1053242, name]
        Observation 3: the astrophysical journal, the atmosphere journal
        Plan 4: The question is asking the published venue of a paper (Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars). We get the answer ———— the astrophysical journal, the atmosphere journal
        Thought 4: The name of the published venues are the astrophysical journal and the atmosphere journal
        Action 4: Finish[the astrophysical journal, the atmosphere journal]
        """,
        "amazon": """Definition of the graph: There are two types of nodes in the graph: item and brand. 
            Item nodes have features: title, description, price, img, category. Brand nodes have features: name. 
            Item nodes are linked to their brand nodes, also_viewed_item nodes, buy_after_viewing_item nodes, also_bought_item nodes, bought_together_item nodes. Brand nodes are linked to their item nodes.
            Question: What is the price of the item Purple Sequin Tiny Dancer Tutu Ballet Dance Fairy Princess Costume Accessory?
            Plan 1: The question is asking the price of a item node (Purple Sequin Tiny Dancer Tutu Ballet Dance Fairy Princess Costume Accessory). Thus, we need to locate the relevant item and extract the price from the graph.
            Thought 1: We need to find the node in the graph.
            Action 1: Retrieve[Purple Sequin Tiny Dancer Tutu Ballet Dance Fairy Princess Costume Accessory]
            Observation 1: The ID of this node is 0000037214.
            Plan 2: The question is asking the price of a item node (Purple Sequin Tiny Dancer Tutu Ballet Dance Fairy Princess Costume Accessory). We have the item node. Thus, we need to extract the price from the graph.
            Thought 2: We need to check the node feature (price) from the graph.
            Action 2: Feature[0000037214, price]
            Observation 2: 6.99
            Plan 3: The question is asking the price of a item node (Purple Sequin Tiny Dancer Tutu Ballet Dance Fairy Princess Costume Accessory). We get the answer ———— 6.99.
            Thought 3: The price of the item is 6.99.
            Action 3: Finish[6.99]
            
            Definition of the graph: There are two types of nodes in the graph: item and brand. 
            Item nodes have features: title, description, price, img, category. Brand nodes have features: name. 
            Item nodes are linked to their brand nodes, also_viewed_item nodes, buy_after_viewing_item nodes, also_bought_item nodes, bought_together_item nodes. Brand nodes are linked to their item nodes.
            Question: How many also_viewed_item does item The Sherlock Holmes Audio Collection have?
            Plan 1: The question is asking the number of also_viewed_item neighbour of a item node (The Sherlock Holmes Audio Collection). Thus, We need to find the node in the graph and then count the node's also_viewed_item neighbor.
            Thought 1: We need to find the node in the graph.
            Action 1: Retrieve[The Sherlock Holmes Audio Collection]
            Observation 1: The ID of this node is 0001048236.
            Plan 2: The question is asking the number of also_viewed_item neighbour of a item node (The Sherlock Holmes Audio Collection). We have the item node. Thus, We need to count the node's also_viewed_item neighbor.
            Thought 2: We need to calculate the node's also_viewed_item neighbor degree from the graph.
            Action 2: Degree[0001048236, also_viewed_item]
            Observation 2: 9
            Plan 3: The question is asking the number of also_viewed_item neighbour of a item node (The Sherlock Holmes Audio Collection). We get the answer ———— 9.
            Thought 3: The number of also_viewed_item neighbors is 9.
            Action 3: Finish[9]
            
            Definition of the graph: There are two types of nodes in the graph: item and brand. 
            Item nodes have features: title, description, price, img, category. Brand nodes have features: name. 
            Item nodes are linked to their brand nodes, also_viewed_item nodes, buy_after_viewing_item nodes, also_bought_item nodes, bought_together_item nodes. Brand nodes are linked to their item nodes.
            Question: What is the brand of item Dr. Suess 19163 Dr. Seuss Puzzle 3 Pack Bundle?
            Plan 1: The question is asking brand of a item node (Dr. Suess 19163 Dr. Seuss Puzzle 3 Pack Bundle). We need to find the node and get its brand in the graph.
            Thought 1: We need to find the node in the graph.
            Action 1: Retrieve[Dr. Suess 19163 Dr. Seuss Puzzle 3 Pack Bundle]
            Observation 1: The ID of this node is 0000191639.
            Plan 2: The question is asking brand of a item node (Dr. Suess 19163 Dr. Seuss Puzzle 3 Pack Bundle). We have the item node. Thus, we need to get its brand in the graph.
            Thought 2: We need to check the node's brand neighbor from the graph.
            Action 2: Neighbor[0000191639, brand]
            Observation 2: ['brand_5']
            Plan 3: The question is asking brand of a item node (Dr. Suess 19163 Dr. Seuss Puzzle 3 Pack Bundle). We have the brand node. Thus we need to get the name of the brand.
            Thought 3: The ID of this brand node is brand_5. We need to check the brand node's feature (name) from the graph.
            Action 3: Feature[brand_5, name]
            Observation 3: Dr. Seuss.
            Plan 4: The question is asking brand of a item node (Dr. Suess 19163 Dr. Seuss Puzzle 3 Pack Bundle). We get the answer ———— Dr. Seuss.
            Thought 4: The brand name of the item is Dr. Seuss.
            Action 4: Finish[Dr. Seuss]
        """,
        "biomedical": """Definition of the graph: There are eleven types of nodes in the graph: Anatomy, Biological Process, Cellular Component, Compound, Disease, Gene, Molecular Function, Pathway, Pharmacologic Class, Side Effect, Symptom.
        Each node has name feature.
        There are these types of edges: Anatomy-downregulates-Gene, Anatomy-expresses-Gene, Anatomy-upregulates-Gene, Compound-binds-Gene, Compound-causes-Side Effect, Compound-downregulates-Gene, Compound-palliates-Disease, Compound-resembles-Compound, Compound-treats-Disease, Compound-upregulates-Gene, Disease-associates-Gene, Disease-downregulates-Gene, Disease-localizes-Anatomy, Disease-presents-Symptom, Disease-resembles-Disease, Disease-upregulates-Gene, Gene-covaries-Gene, Gene-interacts-Gene, Gene-participates-Biological Process, Gene-participates-Cellular Component, Gene-participates-Molecular Function, Gene-participates-Pathway, Gene-regulates-Gene, Pharmacologic Class-includes-Compound.
        Question: What compounds can be used to treat Crohn's disease? Please answer the compound names rather than IDs.
        Plan 1: The question is asking the compounds which can be used to treat a disease (Crohn's disease). We need to find the node in the graph and then find the compounds which can be used to treat this disease.
        Thought 1: We need to find the node in the graph.
        Action 1: Retrieve[Crohn's disease]
        Observation 1: The ID of this node is DOID:8778.
        Plan 2: The question is asking the compounds which can be used to treat a disease (Crohn's disease). We have the node of the disease. Thus, we need to find the compounds which can be used to treat this disease.
        Thought 2: We need to check the node's 'Compound-treats-Disease' neighbor from the graph.
        Action 2: Neighbor[DOID:8778, Compound-treats-Disease]
        Observation 2: ['DB01014', 'DB00244', 'DB00795', 'DB00993', 'DB00635', 'DB01033']
        Plan 3: The question is asking the compounds which can be used to treat a disease (Crohn's disease). We have the nodes of the compounds. Thus, we need to get their names.
        Thought 3: The IDs of the compounds are 'DB01014', 'DB00244', 'DB00795', 'DB00993', 'DB00635', 'DB01033'. We need to get their names.
        Action 3: Feature[DB01014, name], Feature[DB00244, name], Feature[DB00795, name], Feature[DB00993, name], Feature[DB00635, name], Feature[DB01033, name]
        Observation 3: Balsalazide, Balsalazide, Mesalazine, Sulfasalazine, Azathioprine, Prednisone, Mercaptopurine
        Plan 4: The question is asking the compounds which can be used to treat a disease (Crohn's disease). We get the answer ———— Balsalazide, Mesalazine, Sulfasalazine, Azathioprine, Prednisone, Mercaptopurine.
        Thought 4: The name of compounds are Balsalazide, Mesalazine, Sulfasalazine, Azathioprine, Prednisone, Mercaptopurine.
        Action 4: Finish[Balsalazide, Mesalazine, Sulfasalazine, Azathioprine, Prednisone, Mercaptopurine]
        
        Definition of the graph: There are eleven types of nodes in the graph: Anatomy, Biological Process, Cellular Component, Compound, Disease, Gene, Molecular Function, Pathway, Pharmacologic Class, Side Effect, Symptom.
        Each node has name feature.
        There are these types of edges: Anatomy-downregulates-Gene, Anatomy-expresses-Gene, Anatomy-upregulates-Gene, Compound-binds-Gene, Compound-causes-Side Effect, Compound-downregulates-Gene, Compound-palliates-Disease, Compound-resembles-Compound, Compound-treats-Disease, Compound-upregulates-Gene, Disease-associates-Gene, Disease-downregulates-Gene, Disease-localizes-Anatomy, Disease-presents-Symptom, Disease-resembles-Disease, Disease-upregulates-Gene, Gene-covaries-Gene, Gene-interacts-Gene, Gene-participates-Biological Process, Gene-participates-Cellular Component, Gene-participates-Molecular Function, Gene-participates-Pathway, Gene-regulates-Gene, Pharmacologic Class-includes-Compound.
        Question: What is the inchikey of Caffeine?
        Plan 1: The question is asking the inchikey of a compound node (Caffeine). We need to find the node in the graph and then get its inchikey.
        Thought 1: We need to find the node in the graph.
        Action 1: Retrieve[Caffeine]
        Observation 1: The ID of this node is DB00201.
        Plan 2: The question is asking the inchikey of a compound node (Caffeine). We have the node. Thus, we need to get its inchikey.
        Thought 2: We need to check the node's 'inchikey' feature from the graph.
        Action 2: Feature[DB00201, inchikey]
        Observation 2: InChIKey=RYYVLZVUVIJVGH-UHFFFAOYSA-N
        Plan 3: The question is asking the inchikey of a compound node (Caffeine). We get the answer ———— InChIKey=RYYVLZVUVIJVGH-UHFFFAOYSA-N.
        Thought 3: The inchikey of the node is InChIKey=RYYVLZVUVIJVGH-UHFFFAOYSA-N.
        Action 3: Finish[InChIKey=RYYVLZVUVIJVGH-UHFFFAOYSA-N]
        
        Definition of the graph: There are eleven types of nodes in the graph: Anatomy, Biological Process, Cellular Component, Compound, Disease, Gene, Molecular Function, Pathway, Pharmacologic Class, Side Effect, Symptom.
        Each node has name feature.
        There are these types of edges: Anatomy-downregulates-Gene, Anatomy-expresses-Gene, Anatomy-upregulates-Gene, Compound-binds-Gene, Compound-causes-Side Effect, Compound-downregulates-Gene, Compound-palliates-Disease, Compound-resembles-Compound, Compound-treats-Disease, Compound-upregulates-Gene, Disease-associates-Gene, Disease-downregulates-Gene, Disease-localizes-Anatomy, Disease-presents-Symptom, Disease-resembles-Disease, Disease-upregulates-Gene, Gene-covaries-Gene, Gene-interacts-Gene, Gene-participates-Biological Process, Gene-participates-Cellular Component, Gene-participates-Molecular Function, Gene-participates-Pathway, Gene-regulates-Gene, Pharmacologic Class-includes-Compound.
        Question: How many side effects does Caffeine have?
        Plan 1: The question is asking the number of side effects of a compound node (Caffeine). We need to find the node in the graph and then get the number of the node's Compound-causes-Side Effect.
        Thought 1: We need to find the node in the graph.
        Action 1: Retrieve[Caffeine]
        Observation 1: The ID of this node is DB00201.
        Plan 2: The question is asking the number of side effects of a compound node (Caffeine). We get the node. Thus, we need to get the number of the node's Compound-causes-Side Effect.
        Thought 2: We need to calculate the number of the node's 'Compound-causes-Side Effect' neighbors from the graph.
        Action 2: Degree[DB00201, 'Compound-causes-Side Effect']
        Observation 2: 58
        Plan 3: The question is asking the number of side effects of a compound node (Caffeine). We get the answer ———— 58.
        Thought 3: The number of 'Compound-causes-Side Effect' neighbors are 58.
        Action 3: Finish[58]
        """,
        "legal": """Definition of the graph: There are four types of nodes in the graph: opinion, opinion_cluster, docket, and court.
        Opinion nodes have features: plain_text. Opinion_cluster nodes have features: syllabus, judges, case_name, attorneys. Docket nodes have features: pacer_case_id, case_name. Court nodes have features: full_name, start_date, end_date, citation_string.
        Opinion nodes are linked to their reference nodes and cited_by nodes, as well as their opinion_cluster nodes. Opinion_cluster nodes are linked to opinion nodes and docket nodes. Docket nodes are linked to opinion_cluster nodes and court nodes. Court nodes are linked to docket nodes.
        Question: Who are the judges that involved in the opinion_cluster: <p>Constitutional law — Title of act — Mortgage—Deed—Liability of grantee — Act of June IB, 1878, P. L. BOB.</p> <p>1. The Act of June 12, 1878, P. L. 205, entitled “An Act to define the liability of grantees of real estate for the incumbrances thereon,” is sufficiently comprehensive in its title to cover both the first and second sections of the act, and does not violate art. Ill, sec. 3 of the constitution of Pennsylvania.</p> <p>2. Where an owner of land creates two mortgages thereon and then conveys the property subject to the mortgages, the grantee covenanting in the deed itself to assume and pay the mortgages as a part of the consideration, and thereafter the grantee conveys the land to another person without expressly assuming any continuing liability to pay the mortgages, such grantee is not liable to the mortgag^ for payment of the mortgages. £V</p>?
        Plan 1: The question is asking the judges of this opinion_cluster node (<p>Constitutional law — Title of act — Mortgage—Deed—Liability of grantee — Act of June IB, 1878, P. L. BOB.</p> <p>1. The Act of June 12, 1878, P. L. 205, entitled “An Act to define the liability of grantees of real estate for the incumbrances thereon,” is sufficiently comprehensive in its title to cover both the first and second sections of the act, and does not violate art. Ill, sec. 3 of the constitution of Pennsylvania.</p> <p>2. Where an owner of land creates two mortgages thereon and then conveys the property subject to the mortgages, the grantee covenanting in the deed itself to assume and pay the mortgages as a part of the consideration, and thereafter the grantee conveys the land to another person without expressly assuming any continuing liability to pay the mortgages, such grantee is not liable to the mortgag^ for payment of the mortgages. £V</p>). We need to find the node in the graph and then get the judges.
        Thought 1: We need to find the node in the graph.
        Action 1: Retrieve[<p>Constitutional law — Title of act — Mortgage—Deed—Liability of grantee — Act of June IB, 1878, P. L. BOB.</p> <p>1. The Act of June 12, 1878, P. L. 205, entitled “An Act to define the liability of grantees of real estate for the incumbrances thereon,” is sufficiently comprehensive in its title to cover both the first and second sections of the act, and does not violate art. Ill, sec. 3 of the constitution of Pennsylvania.</p> <p>2. Where an owner of land creates two mortgages thereon and then conveys the property subject to the mortgages, the grantee covenanting in the deed itself to assume and pay the mortgages as a part of the consideration, and thereafter the grantee conveys the land to another person without expressly assuming any continuing liability to pay the mortgages, such grantee is not liable to the mortgag^ for payment of the mortgages. £V</p>]
        Observation 1: The ID of this node is opc-6381448.
        Plan 2: The question is asking the judges of this opinion_cluster node (<p>Constitutional law — Title of act — Mortgage—Deed—Liability of grantee — Act of June IB, 1878, P. L. BOB.</p> <p>1. The Act of June 12, 1878, P. L. 205, entitled “An Act to define the liability of grantees of real estate for the incumbrances thereon,” is sufficiently comprehensive in its title to cover both the first and second sections of the act, and does not violate art. Ill, sec. 3 of the constitution of Pennsylvania.</p> <p>2. Where an owner of land creates two mortgages thereon and then conveys the property subject to the mortgages, the grantee covenanting in the deed itself to assume and pay the mortgages as a part of the consideration, and thereafter the grantee conveys the land to another person without expressly assuming any continuing liability to pay the mortgages, such grantee is not liable to the mortgag^ for payment of the mortgages. £V</p>). We have the node. Thus, we need to get the judges.
        Thought 2: We need to check the node's 'judges' feature from the graph.
        Action 2: Feature[opc-6381448, judges]
        Observation 2: Brown, Elkin, Fell, Mestrezat, Moschzisker, Potter, Stewart
        Plan 3: The question is asking the judges of this opinion_cluster node (<p>Constitutional law — Title of act — Mortgage—Deed—Liability of grantee — Act of June IB, 1878, P. L. BOB.</p> <p>1. The Act of June 12, 1878, P. L. 205, entitled “An Act to define the liability of grantees of real estate for the incumbrances thereon,” is sufficiently comprehensive in its title to cover both the first and second sections of the act, and does not violate art. Ill, sec. 3 of the constitution of Pennsylvania.</p> <p>2. Where an owner of land creates two mortgages thereon and then conveys the property subject to the mortgages, the grantee covenanting in the deed itself to assume and pay the mortgages as a part of the consideration, and thereafter the grantee conveys the land to another person without expressly assuming any continuing liability to pay the mortgages, such grantee is not liable to the mortgag^ for payment of the mortgages. £V</p>). We get the answer ———— Brown, Elkin, Fell, Mestrezat, Moschzisker, Potter, Stewart.
        Thought 3: The judges of this opinion cluster is Brown, Elkin, Fell, Mestrezat, Moschzisker, Potter, Stewart
        Action 3: Finish[Brown, Elkin, Fell, Mestrezat, Moschzisker, Potter, Stewart]

        Definition of the graph: There are four types of nodes in the graph: opinion, opinion_cluster, docket, and court.
        Opinion nodes have features: plain_text. Opinion_cluster nodes have features: syllabus, judges, case_name, attorneys. Docket nodes have features: pacer_case_id, case_name. Court nodes have features: full_name, start_date, end_date, citation_string.
        Opinion nodes are linked to their reference nodes and cited_by nodes, as well as their opinion_cluster nodes. Opinion_cluster nodes are linked to opinion nodes and docket nodes. Docket nodes are linked to opinion_cluster nodes and court nodes. Court nodes are linked to docket nodes.
        Question: How many opinions are contained in this opinion_cluster: <p>A general statement that the decision of a state court is against the constitutional rights of the objecting party, or against the Fourteenth Amendment, or that it is without due process of law,-particularly when these objections appear only in specifications of error, so called, will not raise a Federal question, even where the judgment is a final one within Rev. Stat. § 709.</p> <p>In these cases there was no final judgment, such as is provided for in Rev. Stat. § 709, and there does not appear to have arisen any Federal question whatever.</p>'},
        Plan 1: The question is asking the number of the opinion neighbors of this node (<p>A general statement that the decision of a state court is against the constitutional rights of the objecting party, or against the Fourteenth Amendment, or that it is without due process of law,-particularly when these objections appear only in specifications of error, so called, will not raise a Federal question, even where the judgment is a final one within Rev. Stat. § 709.</p> <p>In these cases there was no final judgment, such as is provided for in Rev. Stat. § 709, and there does not appear to have arisen any Federal question whatever.</p>). We need to find the node in the graph and count its opinion neighbors.
        Thought 1: We need to find the node in the graph.
        Action 1: Retrieve[<p>A general statement that the decision of a state court is against the constitutional rights of the objecting party, or against the Fourteenth Amendment, or that it is without due process of law,-particularly when these objections appear only in specifications of error, so called, will not raise a Federal question, even where the judgment is a final one within Rev. Stat. § 709.</p> <p>In these cases there was no final judgment, such as is provided for in Rev. Stat. § 709, and there does not appear to have arisen any Federal question whatever.</p>]
        Observation 1: The ID of this node is opc-94592.
        Plan 2: The question is asking the number of the opinion neighbors of this node (<p>A general statement that the decision of a state court is against the constitutional rights of the objecting party, or against the Fourteenth Amendment, or that it is without due process of law,-particularly when these objections appear only in specifications of error, so called, will not raise a Federal question, even where the judgment is a final one within Rev. Stat. § 709.</p> <p>In these cases there was no final judgment, such as is provided for in Rev. Stat. § 709, and there does not appear to have arisen any Federal question whatever.</p>). We have the node. Thus, we need to count its opinion neighbors.
        Thought 2: We need to calculate the node's "opinion" neighbor degree from the graph.
        Action 2: Degree[opc-94592, opinion]
        Observation 2: 1
        Plan 3: The question is asking the number of the opinion neighbors of this node (<p>A general statement that the decision of a state court is against the constitutional rights of the objecting party, or against the Fourteenth Amendment, or that it is without due process of law,-particularly when these objections appear only in specifications of error, so called, will not raise a Federal question, even where the judgment is a final one within Rev. Stat. § 709.</p> <p>In these cases there was no final judgment, such as is provided for in Rev. Stat. § 709, and there does not appear to have arisen any Federal question whatever.</p>). We get the answer ———— 1
        Thought 3: The number of the opinion neighbors is 1
        Action 3: Finish[1]

        Definition of the graph: There are four types of nodes in the graph: opinion, opinion_cluster, docket, and court.
        Opinion nodes have features: plain_text. Opinion_cluster nodes have features: syllabus, judges, case_name, attorneys. Docket nodes have features: pacer_case_id, case_name. Court nodes have features: full_name, start_date, end_date, citation_string.
        Opinion nodes are linked to their reference nodes and cited_by nodes, as well as their opinion_cluster nodes. Opinion_cluster nodes are linked to opinion nodes and docket nodes. Docket nodes are linked to opinion_cluster nodes and court nodes. Court nodes are linked to docket nodes.
        Question: What is the docket pacer case ID for this opinion cluster: In May, 1873, a brewer pays to the collector $100 as special tax, for which a special-tax stamp is issued to Mm. Subsequently he shoivs to. the satisfaction of the Commissions)\' of Internal Revenue that he was not a brewer of more than 500 barrels, and henee that he is entitled to have $50 refunded. The Commissioner issues his certificate for a special-tax stamp of the value of%50 returned. The Treasury refuses payment of the certificate. The breioe)\' brings his action to recover the amount thereof.</p> <p>I. The Commissioner of Internal Revenue being authorized by one section of the statutes (Rev. Stat., § 3220) to refund taxes “unjustly assessed or excessive in, amount," and by another, (§ 3426,) the value of special-tax stamps ‘‘where the rates of duties represented thereby have been paid in error or remitted," it is immaterial whether his certificate be in form under the one section or the other so long as it represents a case where the Commissioner is authorized to refund and the amoirnt which the party is entitled to have refunded.</p> <p>II. When a controversy under the revenue laws is by law required to be determined by other officers and other tribunals, this court is without jurisdiction; but where the proper officer has determined a question, and awarded an allowance, and filed a certificate, and exhausted his jurisdiction, and the Treasury has refused to carry out the award by mating payment, this court has jurisdiction of an action brought to recover the. award..
        Plan 1: The question is asking the judges of this opinion cluster node (In May, 1873, a brewer pays to the collector $100 as special tax, for which a special-tax stamp is issued to Mm. Subsequently he shoivs to. the satisfaction of the Commissions)\' of Internal Revenue that he was not a brewer of more than 500 barrels, and henee that he is entitled to have $50 refunded. The Commissioner issues his certificate for a special-tax stamp of the value of%50 returned. The Treasury refuses payment of the certificate. The breioe)\' brings his action to recover the amount thereof.</p> <p>I. The Commissioner of Internal Revenue being authorized by one section of the statutes (Rev. Stat., § 3220) to refund taxes “unjustly assessed or excessive in, amount," and by another, (§ 3426,) the value of special-tax stamps ‘‘where the rates of duties represented thereby have been paid in error or remitted," it is immaterial whether his certificate be in form under the one section or the other so long as it represents a case where the Commissioner is authorized to refund and the amoirnt which the party is entitled to have refunded.</p> <p>II. When a controversy under the revenue laws is by law required to be determined by other officers and other tribunals, this court is without jurisdiction; but where the proper officer has determined a question, and awarded an allowance, and filed a certificate, and exhausted his jurisdiction, and the Treasury has refused to carry out the award by mating payment, this court has jurisdiction of an action brought to recover the. award.). We need to find the node in the graph and get its judges.
        Thought 1: We need to find the node in the graph.
        Action 1: Retrieve[In May, 1873, a brewer pays to the collector $100 as special tax, for which a special-tax stamp is issued to Mm. Subsequently he shoivs to. the satisfaction of the Commissions)\' of Internal Revenue that he was not a brewer of more than 500 barrels, and henee that he is entitled to have $50 refunded. The Commissioner issues his certificate for a special-tax stamp of the value of%50 returned. The Treasury refuses payment of the certificate. The breioe)\' brings his action to recover the amount thereof.</p> <p>I. The Commissioner of Internal Revenue being authorized by one section of the statutes (Rev. Stat., § 3220) to refund taxes “unjustly assessed or excessive in, amount," and by another, (§ 3426,) the value of special-tax stamps ‘‘where the rates of duties represented thereby have been paid in error or remitted," it is immaterial whether his certificate be in form under the one section or the other so long as it represents a case where the Commissioner is authorized to refund and the amoirnt which the party is entitled to have refunded.</p> <p>II. When a controversy under the revenue laws is by law required to be determined by other officers and other tribunals, this court is without jurisdiction; but where the proper officer has determined a question, and awarded an allowance, and filed a certificate, and exhausted his jurisdiction, and the Treasury has refused to carry out the award by mating payment, this court has jurisdiction of an action brought to recover the. award.]
        Observation 1: The ID of this node is opc-8599951.
        Plan 2: The question is asking the judges of this opinion cluster node (In May, 1873, a brewer pays to the collector $100 as special tax, for which a special-tax stamp is issued to Mm. Subsequently he shoivs to. the satisfaction of the Commissions)\' of Internal Revenue that he was not a brewer of more than 500 barrels, and henee that he is entitled to have $50 refunded. The Commissioner issues his certificate for a special-tax stamp of the value of%50 returned. The Treasury refuses payment of the certificate. The breioe)\' brings his action to recover the amount thereof.</p> <p>I. The Commissioner of Internal Revenue being authorized by one section of the statutes (Rev. Stat., § 3220) to refund taxes “unjustly assessed or excessive in, amount," and by another, (§ 3426,) the value of special-tax stamps ‘‘where the rates of duties represented thereby have been paid in error or remitted," it is immaterial whether his certificate be in form under the one section or the other so long as it represents a case where the Commissioner is authorized to refund and the amoirnt which the party is entitled to have refunded.</p> <p>II. When a controversy under the revenue laws is by law required to be determined by other officers and other tribunals, this court is without jurisdiction; but where the proper officer has determined a question, and awarded an allowance, and filed a certificate, and exhausted his jurisdiction, and the Treasury has refused to carry out the award by mating payment, this court has jurisdiction of an action brought to recover the. award.). We have the node. Thus, we need to get its judges.
        Thought 2: We need to check the node's 'judges' feature from the graph.
        Action 2: Feature[opc-8599951]
        Observation 2: {'judges': 'Eichaedson', 'case_name': 'Kaufman v. United States', 'attorneys': 'Mr. P. P. Pye for the claimant., Mr. Assistant Attorney-General Simons for the defendants.', 'syllabus': '<p>In May, 1873, a brewer pays to the collector $100 as special tax, for which a special-tax stamp is issued to Mm. Subsequently he shoivs to. the satisfaction of the Commissions)\' of Internal Revenue that he was not a brewer of more than 500 barrels, and henee that he is entitled to have $50 refunded. The Commissioner issues his certificate for a special-tax stamp of the value of%50 returned. The Treasury refuses payment of the certificate. The breioe)\' brings his action to recover the amount thereof.</p> <p>I. The Commissioner of Internal Revenue being authorized by one section of the statutes (Rev. Stat., § 3220) to refund taxes “unjustly assessed or excessive in, amount," and by another, (§ 3426,) the value of special-tax stamps ‘‘where the rates of duties represented thereby have been paid in error or remitted," it is immaterial whether his certificate be in form under the one section or the other so long as it represents a case where the Commissioner is authorized to refund and the amoirnt which the party is entitled to have refunded.</p> <p>II. When a controversy under the revenue laws is by law required to be determined by other officers and other tribunals, this court is without jurisdiction; but where the proper officer has determined a question, and awarded an allowance, and filed a certificate, and exhausted his jurisdiction, and the Treasury has refused to carry out the award by mating payment, this court has jurisdiction of an action brought to recover the. award.</p>'}
        Plan 3: The question is asking the judges of this opinion cluster node (In May, 1873, a brewer pays to the collector $100 as special tax, for which a special-tax stamp is issued to Mm. Subsequently he shoivs to. the satisfaction of the Commissions)\' of Internal Revenue that he was not a brewer of more than 500 barrels, and henee that he is entitled to have $50 refunded. The Commissioner issues his certificate for a special-tax stamp of the value of%50 returned. The Treasury refuses payment of the certificate. The breioe)\' brings his action to recover the amount thereof.</p> <p>I. The Commissioner of Internal Revenue being authorized by one section of the statutes (Rev. Stat., § 3220) to refund taxes “unjustly assessed or excessive in, amount," and by another, (§ 3426,) the value of special-tax stamps ‘‘where the rates of duties represented thereby have been paid in error or remitted," it is immaterial whether his certificate be in form under the one section or the other so long as it represents a case where the Commissioner is authorized to refund and the amoirnt which the party is entitled to have refunded.</p> <p>II. When a controversy under the revenue laws is by law required to be determined by other officers and other tribunals, this court is without jurisdiction; but where the proper officer has determined a question, and awarded an allowance, and filed a certificate, and exhausted his jurisdiction, and the Treasury has refused to carry out the award by mating payment, this court has jurisdiction of an action brought to recover the. award.). We get the answer ———— Eichaedson.
        Thought 3: The judges of this opinion cluster is Eichaedson
        Action 3: Finish[Eichaedson]
        """,
    "goodreads": """Definition of the graph: There are four types of nodes in the graph: book, author, publisher, and series. 
        Book nodes have features: country_code, language_code, is_ebook, title, description, format, num_pages, publication_year, url, popular_shelves, and genres. Author nodes have features: name. Publisher nodes have features: name. Series nodes have features: title and description. 
        Book nodes are linked to their author nodes, publisher nodes, series nodes and similar_books nodes. Author nodes are linked to their book nodes. Publisher nodes are linked to their book nodes. Series nodes are linked to their book nodes.
        Question: Who is the author of the book 'The Great Gatsby'?
        Plan 1: The question is asking for the author of a specific book (The Great Gatsby). We need to find the book node and then get its author.
        Thought 1: We need to find the book node in the graph.
        Action 1: Retrieve[The Great Gatsby]
        Observation 1: The ID of this book node is 21366268.
        Plan 2: The question is asking for the author of a specific book (The Great Gatsby). We have the book node. Thus, we need to get its author.
        Thought 2: We need to check the book node's author neighbor from the graph.
        Action 2: Neighbor[21366268, author]
        Observation 2: ['3190']
        Plan 3: The question is asking for the author of a specific book (The Great Gatsby). We have the author node. Thus, we need to get its name.
        Thought 3: The ID of the author is 3190. We need to check the name.
        Action 3: Feature[3190, name]
        Observation 3: F. Scott Fitzgerald
        Plan 4: The question is asking for the author of a specific book (The Great Gatsby). We get the answer ———— F. Scott Fitzgerald
        Thought 4: The author of 'The Great Gatsby' is F. Scott Fitzgerald.
        Action 4: Finish[F. Scott Fitzgerald]
        
        Definition of the graph: There are four types of nodes in the graph: book, author, publisher, and series. 
        Book nodes have features: country_code, language_code, is_ebook, title, description, format, num_pages, publication_year, url, popular_shelves, and genres. Author nodes have features: name. Publisher nodes have features: name. Series nodes have features: title and description. 
        Book nodes are linked to their author nodes, publisher nodes, series nodes and similar_books nodes. Author nodes are linked to their book nodes. Publisher nodes are linked to their book nodes. Series nodes are linked to their book nodes.
        Question: What is the genre of the book '1984'?
        Plan 1: The question is asking for the genre of a specific book (1984). We need to find the book node and then get its genre.
        Thought 1: We need to find the book node in the graph.
        Action 1: Retrieve[1984]
        Observation 1: The ID of this book node is 3475269.
        Plan 2: The question is asking for the genre of a specific book (1984). We have the book node. Thus, we need to get its genre.
        Thought 2: We need to check the book node's genre feature from the graph.
        Action 2: Feature[3475269, genres]
        Observation 2: currently-reading, classics, fiction
        Plan 3: The question is asking for the genre of a specific book (1984). We have the answer ———— currently-reading, classics, fiction.
        Thought 3: The genres of the book '1984' are currently-reading, classics, fiction.
        Action 3: Finish[currently-reading, classics, fiction]
        
        Definition of the graph: There are four types of nodes in the graph: book, author, publisher, and series. 
        Book nodes have features: country_code, language_code, is_ebook, title, description, format, num_pages, publication_year, url, popular_shelves, and genres. Author nodes have features: name. Publisher nodes have features: name. Series nodes have features: title and description. 
        Book nodes are linked to their author nodes, publisher nodes, series nodes and similar_books nodes. Author nodes are linked to their book nodes. Publisher nodes are linked to their book nodes. Series nodes are linked to their book nodes.
        Question: How many series included the book 'Harry Potter and the Sorcerer's Stone'?
        Plan 1: The question is asking for the number of series included in a specific book (Harry Potter and the Sorcerer's Stone). We need to find the book node and then get its series neighbor degree.
        Thought 1: We need to find the book node for 'Harry Potter and the Sorcerer's Stone' to determine its series.
        Action 1: Retrieve[Harry Potter and the Sorcerer's Stone]
        Observation 1: The ID of this book node is 27421523.
        Plan 2: The question is asking for the number of series included in a specific book (Harry Potter and the Sorcerer's Stone). We have the book node. Thus, we need to get its series neighbor degree.
        Thought 2: We need to calculate the node's series neighbor degree from the graph.
        Action 2: Degree[27421523, series]
        Observation 2: 1
        Plan 3: The question is asking for the number of series included in a specific book (Harry Potter and the Sorcerer's Stone). We have the answer ———— 1.
        Thought 3: The number of series neighbors is 1.
        Action 3: Finish[1]
        """,
    "dblp": """Definition of the graph: There are three types of nodes in the graph: paper, author and venue.
        Paper nodes have features: title, abstract, keywords, lang, and year. Author nodes have features: name and organization. Venue nodes have features: name.
        Paper nodes are linked to their author nodes, venue nodes, reference nodes (the papers this paper cite) and cited_by nodes (other papers which cite this paper). Author nodes are linked to their paper nodes. Venue nodes are linked to their paper nodes.
        Question: What organization is researcher Greg Daville from?
        Plan 1: The question is asking about the organization of a researcher named Greg Daville. We need to find the node of Greg Daville in the graph and check the node's organization feature from the graph.
        Thought 1: We need to find the node corresponding to Greg Daville in the graph.
        Action 1: Retrieve[Greg Daville]
        Observation 1: The ID of this retrieval target node is 53f460a7dabfaee4dc83702a.
        Plan 2: The question is asking about the organization of a researcher named Greg Daville. We have the node of Greg Daville. Thus, we need to check the node's organization feature from the graph.
        Thought 2: We need to check the node's organization feature from the graph.
        Action 2: Feature[53f460a7dabfaee4dc83702a, organization]
        Observation 2: Hove East Sussex, United Kingdom 
        Plan 3: The question is asking about the organization of a researcher named Greg Daville. We have the answer ———— Hove East Sussex, United Kingdom
        Thought 3: The organization of researcher Greg Daville is Hove East Sussex, United Kingdom.
        Action 3: Finish[Hove East Sussex, United Kingdom] 
    
        Definition of the graph: There are three types of nodes in the graph: paper, author and venue.
        Paper nodes have features: title, abstract, keywords, lang, and year. Author nodes have features: name and organization. Venue nodes have features: name.
        Paper nodes are linked to their author nodes, venue nodes, reference nodes (the papers this paper cite) and cited_by nodes (other papers which cite this paper). Author nodes are linked to their paper nodes. Venue nodes are linked to their paper nodes.
        Question: How many papers are written by author Nicholas Lydon?
        Plan 1: The question is asking for the number of papers written by a specific author (Nicholas Lydon). We need to find the author node in the graph and then calculate the number of the author's papers.
        Thought 1: We need to find the author node in the graph.
        Action 1: Retrieve[Nicholas Lydon]
        Observation 1: The ID of this retrieval target node is 53f438c3dabfaedf43596117.
        Plan 2: The question is asking for the number of papers written by a specific author (Nicholas Lydon). We have the anthor node. Thus, we need to calculate the number of the author's papers.
        Thought 2: We need to calculate the "paper" neighbor degree of this node.
        Action 2: Degree[53f438c3dabfaedf43596117, paper]
        Observation 2: 2
        Plan 3: The question is asking for the number of papers written by a specific author (Nicholas Lydon). We have the answer ———— 2
        Thought 3: The number of the paper neighbors is 2
        Action 3: Finish[2]        

        Definition of the graph: There are three types of nodes in the graph: paper, author and venue.
        Paper nodes have features: title, abstract, keywords, lang, and year. Author nodes have features: name and organization. Venue nodes have features: name.
        Paper nodes are linked to their author nodes, venue nodes, reference nodes (the papers this paper cite) and cited_by nodes (other papers which cite this paper). Author nodes are linked to their paper nodes. Venue nodes are linked to their paper nodes.
        Question: What was the publish venue of the paper Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars?
        Plan 1: The question is asking the published venue of a paper (Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars). We need to find the paper node and then get its venue.
        Thought 1: We need to find the node in the graph.
        Action 1: Retrieve[Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars]
        Observation 1: The ID of this node is 2090642949.
        Plan 2: The question is asking the published venue of a paper (Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars). We have the ID of the paper node. Thus, we need to get the venue's name.
        Thought 2: We need to check the node's venue neighbor from the graph.
        Action 2: Neighbor[2090642949, venue]
        Observation 2: ['1980519', '1053242']
        Plan 3: The question is asking the published venue of a paper (Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars). We have the ID of the venue. Thus, we need to get the name of venue.
        Thought 3: The ID of the published venue are 1980519 and 1053242. We need to get their names.
        Action 3: Feature[1980519, name], Feature[1053242, name]
        Observation 3: the astrophysical journal, the atmosphere journal
        Plan 4: The question is asking the published venue of a paper (Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars). We have the answer ———— the astrophysical journal, the atmosphere journal
        Thought 4: The name of the published venues are the astrophysical journal and the atmosphere journal
        Action 4: Finish[the astrophysical journal, the atmosphere journal]
        """,
}

REFLECT_EXAMPLES={
"maple": """Previous trial Details:
Question: Who can be defined as the closest collaborator of author Alice B. Cooper, according to the number of their joint works? Please answer by providing node main feature (e.g., names) rather than node IDs.
Thought 1: The question is asking the closest collaborator of an author (Alice B. Cooper). We need to find the node in the graph.
Action 1: Retrieve[Alice B. Cooper]
Observation 1: The ID of this retrieval target node is 2676686768.
Thought 2: The question is asking the closest collaborator of an author, we need to calculate the "author" neighbor degree of this node to find the collaborators.
Action 2: Degree[2676686768, author]
Observation 2: The node or neighbor type does not exist in the graph. This might because your given neighbor type is not correct. Please modify it.
Thought 3: The question is asking the closest collaborator of an author, we need to check the node's author neighbor from the graph.
Action 3: Neighbor[2676686768, author]
Observation 3: The node or neighbor type does not exist in the graph. This might because your given neighbor type is not correct. Please modify it.
Thought 4: It seems that there is an issue with the graph structure or the given neighbor type.
Action 4: Finish[Unable to retrieve closest collaborator]
Observation 4: Answer is Unable to retrieve closest collaborator

Reflection:
- Recap of the Trial
  - Question: Who can be defined as the closest collaborator of author Alice B. Cooper, according to the number of their joint works? Please answer by providing node main feature (e.g., names) rather than node IDs.
  - Graph Information Used:
    Retrieve[Alice B. Cooper]: To find the node representing Alice B. Cooper.
    Degree[2676686768, author]: To calculate the degree of "author" neighbors.
    Neighbor[2676686768, author]: To list neighbors of type "author"
  - Outcome: The question was not successfully answered because of guessing the wrong answer with Finish[].
- Guided Reflection
  - Understanding the Question
    - Core Goal: Identify the closest collaborator of Alice B. Cooper by evaluating the number of joint works, which requires correctly analyzing the relationships between nodes in the graph.
    - Missed Information: Misinterpreted or overlooked the correct relationship or neighbor type that represents "collaborators."
    - Potential Misunderstandings: Misinterpreted the graph's structure, assuming direct "author" neighbors represent collaborators. The collaboration is mediated by shared "paper" nodes.
  - Analysis of Selected Graph Information
    - Relevance: Retrieve[Alice B. Cooper]: Correct step to locate the node representing the author. Degree and Neighbor with the type "author": Incorrect assumptions about the graph's structure caused these actions to fail.
    - Missed Insights: Did not consider exploring the graph to confirm how collaborators are connected (e.g., through shared papers). Overlooked the use of Feature to extract details about "paper" connections.
    - Redundancies: Repeated attempts to use "author" as the neighbor type without validating its relevance were redundant.
  - Align the Problem Understanding and Graph Function Information
    - Understanding and Selection: The misunderstanding of the question's requirements led to inappropriate graph function calls.
    - Inconsistencies: Assumed collaborators were directly connected as "author" neighbors, which was inconsistent with the graph's definition.
    - Adjustments: Start by understanding the graph's structure and how nodes are linked (e.g., "author" nodes are linked through "paper" nodes). Verify the relationship structure of the graph first using exploratory functions or metadata queries.
- Improved Strategy
  - Updated Understanding of the Problem:
    Collaborators of Alice B. Cooper are determined by shared "paper" nodes. To find the closest collaborator, it is necessary to identify the "paper" neighbors of Alice B. Cooper, then determine the authors with the highest co-authorship count.
  - Revised Graph Selection:
    Use Retrieve[Alice B. Cooper] to find Alice B. Cooper's node.
    Use Neighbor with the type "paper" to list all "paper" neighbors of Alice B. Cooper.
    For each "paper" node, check its "author" neighbors using Neighbor.
    Count the occurrences of each author node in the "paper" neighbors list to find the collaborator with the highest joint works.
    Use Feature to retrieve the name of the closest collaborator.
  - Avoiding Past Issues: 
    Confirm graph structure and neighbor relationships before performing specific queries.
    Avoid redundant actions by verifying assumptions at each step.
    Use appropriate graph functions (e.g., Neighbor and Feature) to extract detailed relationship data.(END OF REFLECTION)\n\n\n\n
    (END OF REFLECTION)
            """,
"amazon": """Previous trial Details:
Question: What is the count of co-viewed items for item Columbia Women's Silver Ridge Lite Long Sleeve Shirt? Please answer by providing node main feature (e.g., names) rather than node IDs.
Thought 1: The question is asking the count of co-viewed items for a item node (Columbia Women's Silver Ridge Lite Long Sleeve Shirt). We need to find the node in the graph.
Action 1: Retrieve[Columbia Women's Silver Ridge Lite Long Sleeve Shirt]
Observation 1: The ID of this retrieval target node is ABCDE00000.
Thought 2: The question is asking the count of co-viewed items for an item, we need to calculate the node's also_viewed_item neighbor degree from the graph.
Action 2: Degree[ABCDE00000, also_viewed_item]
Observation 2: The also_viewed_item neighbor node degree of ABCDE00000 are: 15. 
Thought 3: The question is asking the count of co-viewed items for an item, we need to list the node's also_viewed_item neighbors from the graph.
Action 3: Neighbor[ABCDE00000, also_viewed_item]
Observation 3: The also_viewed_item neighbors of ABCDE00000 are: ['B009XNJPX8', 'B00FAN8CF2', 'B0043EVJJA', 'B009XNJOBQ', 'B0043EVIVE'].
Thought 4:The co-viewed items for item Columbia Women's Silver Ridge Lite Long Sleeve Shirt are 'B009XNJPX8', 'B00FAN8CF2', 'B0043EVJJA', 'B009XNJOBQ', 'B0043EVIVE'.
Action 4:Finish['B009XNJPX8', 'B00FAN8CF2', 'B0043EVJJA', 'B009XNJOBQ', 'B0043EVIVE']
Observation 4: Answer is 'B009XNJPX8', 'B00FAN8CF2', 'B0043EVJJA', 'B009XNJOBQ', 'B0043EVIVE'

Reflection:
- Recap of the Trial
  - Question: What is the count of co-viewed items for the item Columbia Women's Silver Ridge Lite Long Sleeve Shirt? Please answer by providing node main features (e.g., names) rather than node IDs.
  - Graph Information Used:
    Retrieve[Columbia Women's Silver Ridge Lite Long Sleeve Shirt]: To locate the node representing Columbia Women's Silver Ridge Lite Long Sleeve Shirt.
    Degree[ABCDE00000, also_viewed_item]: To calculate the degree of also_viewed_item neighbors for Columbia Women's Silver Ridge Lite Long Sleeve Shirt.
    Neighbor[ABCDE00000, also_viewed_item]: To list neighbors of type also_viewed_item for Columbia Women's Silver Ridge Lite Long Sleeve Shirt.
  - Outcome: The question was not successfully answered because of guessing the wrong answer with Finish[].
- Guided Reflection
  -  Understanding the Question
    - Core Goal: Provide the count of also_viewed_item neighbors for the specified item.
    - Missed Information: Misunderstood the requirement to provide only the count of neighbors, leading to an unnecessary listing of neighbor details.
    - Potential Misunderstandings: Assumed the question required listing the names of also_viewed_item neighbors instead of focusing on the numeric count.
  - Analysis of Selected Graph Information
    - Relevance: Retrieve: Correct step to locate the item node. Degree: Appropriate to calculate the degree of also_viewed_item neighbors, directly addressing the question. Neighbor: Redundant in this context since the question did not require listing the neighbors.
    - Missed Insights: Overlooked that Degree alone could answer the question by providing the required count.
    - Redundancies: The additional step of Neighbor was unnecessary and complicated the process.
  - Align the Problem Understanding and Graph Function Information
    - Understanding and Selection: Misaligned the use of Neighbor, assuming additional details were required.
    - Inconsistencies: Misunderstood the question's requirement, leading to redundant steps.
    - Adjustments: Focus on using Degree to calculate and return the count without additional queries.
- Improved Strategy
  - Updated Understanding of the Problem:
    The question requires the numeric count of also_viewed_item neighbors for the specified item. Listing neighbor details is unnecessary.
  - Revised Graph Selection:
    Use Retrieve[Columbia Women's Silver Ridge Lite Long Sleeve Shirt] to locate the target node.
    Use Degree with the type also_viewed_item to calculate and directly return the count of co-viewed items.
  - Avoiding Past Issues: 
    Focus on the core requirement (providing the count).
    Eliminate redundant queries like Neighbor unless explicitly needed.
    Use concise and relevant graph operations to align with the problem's objective.(END OF REFLECTION)\n\n\n\n
    (END OF REFLECTION)
            """,
"biomedical": """Previous trial Details:
Question: Could you tell me the side effects of the compound Aspirin? Please answer by providing node main feature (e.g., names) rather than node IDs.
Thought 1: The question is related to a compound node (Aspirin). We need to find the node in the graph.
Action 1: Retrieve[Aspirin]
Observation 1: The ID of this retrieval target node is DB00945.
Thought 2: The question is asking the side effects of the compound. We need to check the node's 'Compound-causes-Side Effect' neighbor from the graph.
Action 2: Neighbor[DB00945, Compound-causes-Side Effect]
Observation 2: The Compound-causes-Side Effect neighbors of DB00945 are: ['C1234567'].
Thought 3: The side effect of Aspirin is C1234567.
Action 3: Finish['C1234567']
Observation 3: Answer is 'C1234567'

Reflection: 
- Recap of the Trial
  - Question: What are the side effects of the compound Aspirin? Please answer by providing node main feature (e.g., names) rather than node IDs.
  - Graph Information Used:
    Retrieve[Aspirin]: To locate the node representing Aspirin.
    Neighbor[DB00945, Compound-causes-Side Effect]: To retrieve neighbors of type Compound-causes-Side Effect.
  - Outcome: The question was not successfully answered because of guessing the wrong answer with Finish[].
- Guided Reflection
  -  Understanding the Question
    - Core Goal: Identify and provide the names of all side effects caused by Aspirin using the Compound-causes-Side Effect relationship.
    - Missed Information: Failed to retrieve the name feature of the side effect node.
    - Potential Misunderstandings: Assumed that the side effect node ID was sufficient as an answer, overlooking the need to query and provide the node's name feature.
  - Analysis of Selected Graph Information
    - Relevance: Retrieve: Correct to locate the compound node for Aspirin. Neighbor: Correct to identify the side effect nodes linked to Aspirin.
    - Missed Insights: Did not use Feature to retrieve the name of the side effect node, which is essential to answer the question accurately.
    - Redundancies: None; all actions were directly related to answering the question, but an additional step to fetch the name was required.
  - Align the Problem Understanding and Graph Function Information
    - Understanding and Selection: Misinterpreted the requirement to provide the node's name feature, leading to an incomplete answer.
    - Inconsistencies: Assumed that providing the node ID sufficed, which was inconsistent with the question's explicit requirement.
    - Adjustments: Add a step to query the name feature of each retrieved side effect node.
- Improved Strategy
  - Updated Understanding of the Problem:
    The question requires retrieving the names of all side effects caused by Aspirin.
  - Revised Graph Selection:
    Use Retrieve[Aspirin] to locate the compound node.
    Use Neighbor[DB00945, Compound-causes-Side Effect] to retrieve all side effect nodes.
    Use Feature on each retrieved side effect node to extract their name features.
  - Avoiding Past Issues: 
    Ensure that all answers are provided in terms of the node name feature as required.(END OF REFLECTION)\n\n\n\n
    (END OF REFLECTION)
            """,
"legal": """Previous trial Details:
Question: which court is handling the case listed under the PACER docket number 108735.0? Please answer by providing node main feature (e.g., names) rather than node IDs.
Thought 1: The question is asking for the court handling the case represented by the docket node with the given PACER case ID. We need to find the docket node first.
Action 1: Retrieve[108735.0, Docket]
Observation 1: The ID of this retrieval target node is d-65685551.
Thought 2: Now that we have the docket node, we need to find the associated court node by checking its 'court' neighbor.
Action 2: Neighbor[d-65685551, court]
Observation 2: The court neighbors of d-65685551 are: ['ca9']. 
Thought 3: The court that is handling the case is 'ca9'.
Action 3: Finish[ca9]
Observation 3: Answer is ca9

Reflection: 
- Recap of the Trial
  - Question: Which court is handling the case listed under the PACER docket number 108735.0? Please answer by providing node main feature (e.g., names) rather than node IDs.
  - Graph Information Used:
    Retrieve[108735.0, Docket]: To locate the docket node corresponding to the PACER case ID.
    Neighbor[d-65685551, court]: To retrieve the court node linked to the docket node.
  - Outcome: The question was not successfully answered because of guessing the wrong answer with Finish[].
- Guided Reflection
  -  Understanding the Question
    - Core Goal: Identify the name or main feature of the court handling the case, rather than its node ID.
    - Missed Information: Did not extract the court node's full_name feature, which is the required detail for answering the question correctly.
    - Potential Misunderstandings: Assumed the court node ID (ca9) was sufficient, overlooking the need to query and return the full_name feature.
  - Analysis of Selected Graph Information
    - Relevance: Retrieve: Correctly identified the docket node. Neighbor: Correctly found the court node linked to the docket.
    - Missed Insights: Failed to query the Feature for the full_name of the court node, which was explicitly required to answer the question accurately.
    - Redundancies: None; the steps taken were necessary but incomplete.
  - Align the Problem Understanding and Graph Function Information
    - Understanding and Selection: Correctly identified the relationship between docket and court nodes but failed to retrieve the necessary feature of the court node.
    - Inconsistencies: Provided the node ID (ca9) instead of the required full_name attribute.
    - Adjustments: Add a step to query the Feature of the court node for the full_name attribute.
- Improved Strategy
  - Updated Understanding of the Problem:
    The question requires identifying the court handling the case by providing its full_name attribute, not its node ID.
  - Revised Graph Selection:
    Use Retrieve[108735.0, Docket] to locate the docket node for the PACER case ID.
    Use Neighbor[d-65685551, court] to find the court node linked to the docket.
    Use Feature[ca9, full_name] to retrieve the court's full_name feature.
  - Avoiding Past Issues: 
    Ensure the response includes the requested feature (full_name) rather than raw node IDs.(END OF REFLECTION)\n\n\n\n
    (END OF REFLECTION)

        """,
"goodreads": """Previous trial Details:
Question: What is the most common publication format of books by author Jane Austen? Please answer by providing node main feature (e.g., names) rather than node IDs.
Thought 1: We need to find the author node for Jane Austen to determine the most common publication format of his books.
Action 1: Retrieve[Jane Austen]
Observation 1: The ID of this retrieval target node is author-105.
Thought 2: Now that we have the author node, we need to calculate the node's book neighbor degree from the graph.
Action 2: Degree[author-105, book]
Observation 2: The book neighbor node degree of author-281126 are: 1. 
Thought 3: The number of books is 1.
Action 3: Finish[paperback]
Observation 3: Answer is paperback

Reflection: 
- Recap of the Trial
  - Question: What is the most common publication format of books by author Jane Austen? Please answer by providing node main feature (e.g., names) rather than node IDs.
  - Graph Information Used:
    Retrieve[Jane Austen]: Found the author node.
    Degree[author-105, book]: Checked the degree of book neighbors for the author node.
  - Outcome: The question was not successfully answered because of guessing the wrong answer with Finish[].
- Guided Reflection
  -  Understanding the Question
    - Core Goal: Determine the most common publication format of books written by Jane Austen by analyzing the format feature of the linked book nodes.
    - Missed Information: Failed to retrieve the format feature of the related book nodes.
    - Potential Misunderstandings: Guessed the format (paperback) without extracting data from the graph.
  - Analysis of Selected Graph Information
    - Relevance: Retrieve: Correctly identified the author node. Degree: Checked the number of book nodes but was not sufficient to answer the question.
    - Missed Insights: Did not retrieve the list of book nodes (Neighbor) for further analysis. Failed to query the format feature of the book nodes to determine the most common format.
    - Redundancies: The use of Degree without follow-up queries to analyze the data was unnecessary.
  - Align the Problem Understanding and Graph Function Information
    - Understanding and Selection: Misunderstood the requirement to retrieve and analyze the format feature of books, leading to incomplete reasoning.
    - Inconsistencies: Provided an answer (paperback) without validating it through the graph data.
    - Adjustments: Ensure the retrieval of relevant features (e.g., format) from all related nodes.
- Improved Strategy
  - Updated Understanding of the Problem:
    The question requires identifying the most common format of books written by Jane Austen by analyzing the format feature of all related book nodes.
  - Revised Graph Selection:
    Use Retrieve[Jane Austen] to locate the author node.
    Use Neighbor[author-105, book] to retrieve all book nodes linked to the author.
    Use Feature on each retrieved book node to extract the format feature.
    Analyze the retrieved format values to determine the most common format.
  - Avoiding Past Issues: 
    Do not guess the answer without verifying data through appropriate queries.
    Retrieve and analyze all relevant features of the book nodes.(END OF REFLECTION)\n\n\n\n
    (END OF REFLECTION)


(END OF REFLECTION)
        """,
"dblp": """Previous trial Details:
Question: What is the number of papers that John Doe from Stanford University, Stanford, CA and Jane Smith from Stanford University, Palo Alto, CA co-authored? Please answer by providing node main feature (e.g., names) rather than node IDs.
Thought 1: We need to find the nodes of these two researchers in the graph to determine their co-authored papers
Action 1: Retrieve[John Doe], Retrieve[Jane Smith]
Observation 1: The ID of John Doe's node is author-123456, and the ID of Jane Smith's node is author-789012.
Thought 2: To confirm they are from the same organization, we need to check the organization feature of their nodes.
Action 2: Feature[author-123456, organization], Feature[author-789012, organization]
Observation 2: The organization of author-123456 is Stanford University, Stanford, CA. The organization of author-789012 is Stanford University, Palo Alto, CA.
Thought 3: It's the same organization, now we can calculate the number of papers they co-authored.
Action 3: Degree[author-123456, paper] / Degree[author-789012, paper]
Observation 3: The paper neighbor node degree of author-123456 are: 1. The paper neighbor node degree of author-789012 are: 6. 
Thought 4: There are six papers that they all write.
Action 4: Finish[6]
Observation 4: Answer is 6

Reflection: 
- Recap of the Trial
  - Question: What is the number of papers that John Doe from Stanford University, Stanford, CA and Jane Smith from Stanford University, Palo Alto, CA co-authored? Please answer by providing node main feature (e.g., names) rather than node IDs.
  - Graph Information Used:
    Retrieve[John Doe] and Retrieve[Jane Smith]: Located the nodes for the two authors.
    Feature[author-123456, organization] and Feature[author-789012, organization]: Confirmed they are from the same organization.
    Degree[author-123456, paper] and Degree[author-789012, paper]: Attempted to calculate the number of co-authored papers.
  - Outcome: The question was not successfully answered because of guessing the wrong answer with Finish[].
- Guided Reflection
  -  Understanding the Question
    - Core Goal: Determine the exact number of papers co-authored by John Doe and Jane Smith by finding their common paper neighbors in the graph.
    - Missed Information: Failed to check the Neighbor for both authors to identify common papers.
    - Potential Misunderstandings: Incorrectly interpreted the individual Degree values as the number of co-authored papers.
  - Analysis of Selected Graph Information
    - Relevance: Retrieve: Correctly identified the nodes for the two authors. Feature: Appropriately used to confirm the organization of the authors. Degree: Misused as a proxy for co-authored paper count.
    - Missed Insights: Did not retrieve and compare the actual paper neighbors to identify overlap. Overlooked the need to verify co-authorship explicitly through Neighbor.
    - Redundancies: Checking the node degree for individual authors without retrieving the actual neighbor lists was redundant.
  - Align the Problem Understanding and Graph Function Information
    - Understanding and Selection: The misunderstanding of how node degrees relate to co-authorship caused the failure.
    - Inconsistencies:  Failed to align the goal of finding shared paper nodes with the graph functions available for exploring relationships.
    - Adjustments: Use Neighbor to retrieve the paper neighbors for each author. Compare the retrieved neighbor lists to find overlap, accurately reflecting the number of co-authored papers.
- Improved Strategy
  - Updated Understanding of the Problem:
    The question requires identifying the number of papers co-authored by two authors by comparing their shared paper neighbors.
  - Revised Graph Selection:
    Use Retrieve[John Doe] and Retrieve[Jane Smith] to locate the nodes for the authors.
    Use Feature to confirm their organizational affiliations if needed for context.
    Use Neighbor[author-123456, paper] and Neighbor[author-789012, paper] to retrieve their respective paper neighbors.
    Compare the lists of paper neighbors to identify common nodes (co-authored papers).
    Count the number of common papers and provide the result.
  - Avoiding Past Issues: 
    Avoid relying on Degree as a measure of shared connections without verification.
    Ensure neighbor relationships are explicitly retrieved and analyzed for overlap.(END OF REFLECTION)\n\n\n\n
    (END OF REFLECTION)
        """,
        }


REFLECT_EXAMPLES_SHORT_MULTIPLE={
"maple": """Previous trial:
Question: Who is the closest collaborator of author Alice B. Cooper, based on the number of joint works? Please provide the name, not the node ID.
Thought 1: To find the closest collaborator, we need to retrieve the node for Alice B. Cooper.
Action 1: Retrieve[Alice B. Cooper]
Observation 1: The ID of this retrieval target node is 2676686768.
Thought 2: Next, check the author neighbors of this node to find collaborators.
Action 2: Neighbor[2676686768, author]
Observation 2: The node or neighbor type does not exist in the graph. This might because your given neighbor type is not correct. Please modify it.
Thought 3: The graph structure or neighbor type may be incorrect.
Action 3: Finish[Unable to retrieve closest collaborator]
Observation 3: Answer is Unable to retrieve closest collaborator

Reflection:
- Recap of the Trial
  - Question: Who is the closest collaborator of author Alice B. Cooper, based on the number of joint works? Please provide the name, not the node ID.
  - Graph Information Used:
    Retrieve[Alice B. Cooper]: Retrieved node ID: 2676686768
    Neighbor[2676686768, author]: Error due to incorrect neighbor type.
  - Outcome: Incorrect assumption regarding the graph's neighbor type.
- Guided Reflection
  - Understanding the Question
    - Core Goal: Identify Alice B. Cooper's closest collaborator by determining the collaborator with the most joint works.
    - Missed Information: The neighbor type "author" of author node is not exist. 
    - Misinterpretations: Misinterpreted the graph's structure, assuming direct "author" neighbors represent collaborators. The collaboration is mediated by shared "paper" nodes.
  - Analysis of Selected Graph Information
    - Relevance: Retrieve[Alice B. Cooper]: Correct step to locate the node representing the author.Neighbor with the type "author": Incorrect assumptions about the graph's structure caused these actions to fail.
    - Missed Insights: Did not consider exploring the graph to confirm how collaborators are connected (e.g., through shared papers).
    - Redundancies: Repeated attempts to use "author" as the neighbor type without validating its relevance were redundant.
  - Align the Problem Understanding and Graph Function Information
    -  I assumed the neighbor type was "author" without explicitly confirming it against the graph definition. I should start by understanding the graph's structure and how nodes are linked (e.g., "author" nodes are linked through "paper" nodes).
- Improved Strategy
  - Updated Understanding of the Problem:
    Collaborators of Alice B. Cooper are determined by shared "paper" nodes. To find the closest collaborator, it is necessary to identify the "paper" neighbors of Alice B. Cooper, then determine the authors with the highest co-authorship count.
  - Revised Graph Selection:
    Use Retrieve[Alice B. Cooper] to find Alice B. Cooper's node.
    Use Neighbor with the type "paper" to list all "paper" neighbors of Alice B. Cooper.
    For each "paper" node, check its "author" neighbors using Neighbor.
    Count the occurrences of each author node in the "paper" neighbors list to find the collaborator with the highest joint works.
    Use Feature to retrieve the name of the closest collaborator.
  - Avoiding Past Issues: 
    Confirm graph structure and neighbor relationships before performing specific queries.
    Avoid redundant actions by verifying assumptions at each step.
    Use appropriate graph functions (e.g., Neighbor and Feature) to extract detailed relationship data.(END OF REFLECTION)
            """,
"amazon": """Previous trial:
Question: What is the count of co-viewed items for item Columbia Women's Silver Ridge Lite Long Sleeve Shirt? Please provide the name, not the node ID.
Thought 1: The question asks for co-viewed items for a specific item. First, find the node corresponding to the item.
Action 1: Retrieve[Columbia Women's Silver Ridge Lite Long Sleeve Shirt]
Observation 1: The ID of this retrieval target node is ABCDE00000.
Thought 2: To determine the count of co-viewed items, calculate the also_viewed_item neighbor degree of this node.
Action 2: Degree[ABCDE00000, also_viewed_item]
Observation 2: The degree is 15.
Thought 3: To identify the specific co-viewed items, list the also_viewed_item neighbors for the node.
Action 3: Neighbor[ABCDE00000, also_viewed_item]
Observation 3: Neighbors are ['B009XNJPX8', 'B00FAN8CF2', 'B0043EVJJA', 'B009XNJOBQ', 'B0043EVIVE'].
Thought 4: The co-viewed items for Columbia Women's Silver Ridge Lite Long Sleeve Shirt are those neighbors.
Action 4: Finish['B009XNJPX8', 'B00FAN8CF2', 'B0043EVJJA', 'B009XNJOBQ', 'B0043EVIVE']
Observation 4: Answer is ['B009XNJPX8', 'B00FAN8CF2', 'B0043EVJJA', 'B009XNJOBQ', 'B0043EVIVE'].

Reflection:
- Recap of the Trial
  - Question: What is the count of co-viewed items for the item "Columbia Women's Silver Ridge Lite Long Sleeve Shirt"? Please provide the name, not the node ID.
  - Graph Information Used:
    Retrieve[Columbia Women's Silver Ridge Lite Long Sleeve Shirt]: Used to find the node associated with with the item.
    Degree[ABCDE00000, also_viewed_item]: Used to calculate the count of also_viewed_item neighbors.
    Neighbor[ABCDE00000, also_viewed_item]: Used to retrieve the list of co-viewed item neighbors.
  - Outcome: Correct co-viewed items were retrieved, but redundant steps were taken.
- Guided Reflection
  -  Understanding the Question
    - Core Goal: Identify the count of co-viewed items for the item.
    - Missed Information: The degree alone was sufficient; listing neighbors was unnecessary.
    - Misinterpretations: None, but the approach could be streamlined.
  - Analysis of Selected Graph Information
    - Relevance: Retrieve: Correct for locating the item node. Degree: Appropriate for calculating also_viewed_item degree. Neighbor: Redundant, as listing neighbors wasn't required.
    - Missed Insights: None.
    - Redundancies: Using Neighbor added unnecessary steps for this question.
  - Align the Problem Understanding and Graph Function Information
    - Mixing degree calculation and neighbor listing was inefficient. Focus on using Degree to calculate and return the count without additional queries.
- Improved Strategy
  - Updated Understanding of the Problem:
    The task is to determine the count of co-viewed items using Degree.
  - Revised Graph Selection:
    Use Retrieve[Columbia Women's Silver Ridge Lite Long Sleeve Shirt] to find the target node.
    Use Degree with the type also_viewed_item to get the count of co-viewed items.
  - Avoiding Past Issues: 
    Focus on the core requirement (providing the count).
    Eliminate redundant queries like Neighbor unless explicitly needed.
    Use concise and relevant graph operations to align with the problem's objective.\n\n\n\n
    (END OF REFLECTION)

            """,
"biomedical": """Previous trial:
Question: Could you tell me the side effects of the compound Aspirin? Please provide the name, not the node ID.
Thought 1: The question asks for the side effects of Aspirin. First, find the node for the compound.
Action 1: Retrieve[Aspirin]
Observation 1: The ID of this retrieval target node is DB00945.
Thought 2: To find the side effects, check the 'Compound-causes-Side Effect' neighbors of this node.
Action 2: Neighbor[DB00945, Compound-causes-Side Effect]
Observation 2: The Compound-causes-Side Effect neighbors of DB00945 are: ['C1234567'].
Thought 3: The side effect of Aspirin is C1234567.
Action 3: Finish['C1234567']
Observation 3: Answer is ['C1234567']

Reflection: 
- Recap of the Trial
  - Question: Could you tell me the side effects of the compound Aspirin? Please provide the name, not the node ID.
  - Graph Information Used:
    Retrieve[Aspirin]: To locate the compound node.
    Neighbor[DB00945, Compound-causes-Side Effect]: To identify linked side effect nodes.
  - Outcome: The question was not successfully answered because of guessing the wrong answer with Finish[].
- Guided Reflection
  -  Understanding the Question
    - Core Goal: Provide the names of side effects caused by Aspirin.
    - Missed Information: Did not query the name feature of the side effect node.
    - Misinterpretations: Assumed node IDs sufficed, ignoring the explicit need for names.
  - Analysis of Selected Graph Information
    - Relevance: Retrieve: Correct to locate the compound node for Aspirin. Neighbor: Correct to identify the side effect nodes linked to Aspirin.
    - Missed Insights: Failed to use Feature to extract names.
    - Redundancies: None
  - Align the Problem Understanding and Graph Function Information
    - I misinterpreted the requirement to provide the node's name feature and assumed that providing the node ID sufficed, I should add a step to query the name feature of each retrieved side effect node.
- Improved Strategy
  - Updated Understanding of the Problem:
    The question requires retrieving the names of all side effects caused by Aspirin.
  - Revised Graph Selection:
    Use Retrieve[Aspirin] to locate the compound node.
    Use Neighbor[DB00945, Compound-causes-Side Effect] to retrieve all side effect nodes.
    Use Feature on each retrieved side effect node to extract their name features.
  - Avoiding Past Issues: 
    Ensure that all answers are provided in terms of the node name feature as required.(END OF REFLECTION)\n\n\n\n
    (END OF REFLECTION)

        """,
"legal": """Previous trial:
Question: Which court is handling the case listed under the PACER docket number 108735.0? Please provide the name, not the node ID.
Thought 1: The question asks for the court handling the case. First, find the docket node for the given PACER number.
Action 1: Retrieve[108735.0, Docket]
Observation 1: The ID of this retrieval target node is d-65685551.
Thought 2: To find the court, check the 'court' neighbors of this docket node.
Action 2: Neighbor[d-65685551, court]
Observation 2: The court neighbors of d-65685551 are: ['ca9']. 
Thought 3: The court handling the case is ca9.
Action 3: Finish[ca9]
Observation 3: Answer is ca9

Reflection: 
- Recap of the Trial
  - Question: Which court is handling the case listed under the PACER docket number 108735.0? Please provide the name, not the node ID.
  - Graph Information Used:
    Retrieve[108735.0, Docket]: To locate the docket node.
    Neighbor[d-65685551, court]:  To find the linked court node.
  - Outcome: The question was not successfully answered because of guessing the wrong answer with Finish[].
- Guided Reflection
  -  Understanding the Question
    - Core Goal: Provide the court's name (full_name feature), not the node ID.
    - Missed Information: Failed to query the full_name of the court node.
    - Misinterpretations: Assumed the node ID (ca9) was sufficient.
  - Analysis of Selected Graph Information
    - Relevance: Retrieve: Correctly identified the docket node. Neighbor: Correctly found the court node linked to the docket.
    - Missed Insights: Did not query the court node's full_name.
    - Redundancies: None
  - Align the Problem Understanding and Graph Function Information
    - I provided the node ID (ca9) instead of the required full_name attribute and I should add a step to query the Feature of the court node for the full_name attribute.
- Improved Strategy
  - Updated Understanding of the Problem:
    The task requires retrieving the court's full_name, not its node ID.
  - Revised Graph Selection:
    Use Retrieve[108735.0, Docket] to locate the docket node.
    Use Neighbor[d-65685551, court] to find the court node.
    Use Feature[ca9, full_name] to fetch the court's full_name.
  - Avoiding Past Issues: 
    Ensure the response includes the requested feature (full_name) for accuracy.(END OF REFLECTION)\n\n\n\n
    (END OF REFLECTION)

        """,
"goodreads": """Previous trial:
Question: What is the most common publication format of books by author Jane Austen? Please provide the name, not the node ID.
Thought 1: To find the publication format, first retrieve the author node for Jane Austen.
Action 1: Retrieve[Jane Austen]
Observation 1: The ID of this retrieval target node is author-105.
Thought 2: Calculate the degree of book neighbors for this author node.
Action 2: Degree[author-105, book]
Observation 2: The book neighbor node degree of author-281126 are: 1. 
Thought 3: The most common publication format of Jane Austen's books is paperback.
Action 3: Finish[paperback]
Observation 3: Answer is paperback

Reflection: 
- Recap of the Trial
  - Question: What is the most common publication format of books by author Jane Austen? Please provide the name, not the node ID.
  - Graph Information Used:
    Retrieve[Jane Austen]: Found the author node.
    Degree[author-105, book]: Checked the degree of book neighbors for the author node.
  - Outcome: The answer (paperback) was guessed without retrieving the actual format data.
- Guided Reflection
  -  Understanding the Question
    - Core Goal: Identify the most common publication format of Jane Austen's books by analyzing the format feature of book nodes.
    - Missed Information: Did not query the format feature of linked book nodes.
    - Misinterpretations: Guessed the format without data validation.
  - Analysis of Selected Graph Information
    - Relevance: Retrieve: Correctly identified the author node. Degree: Checked the number of book nodes but was not sufficient to answer the question.
    - Missed Insights: Did not retrieve book nodes (Neighbor) or their format features (Feature).
    - Redundancies: The use of Degree without follow-up queries to analyze the data was unnecessary.
  - Align the Problem Understanding and Graph Function Information
    - I misunderstood the requirement to retrieve and analyze the format feature of books and provided an answer (paperback) without validating it through the graph data. I should ensure the retrieval of relevant features (e.g., format) from all related nodes.
- Improved Strategy
  - Updated Understanding of the Problem:
    The task requires analyzing the format feature of all books linked to Jane Austen to determine the most common format.
  - Revised Graph Selection:
    Use Retrieve[Jane Austen] to locate the author node.
    Use Neighbor[author-105, book] to retrieve all book nodes linked to the author.
    Use Feature on each retrieved book node to extract the format feature.
    Analyze the retrieved format values to determine the most common format.
  - Avoiding Past Issues: 
    Do not guess the answer without verifying data through appropriate queries.
    Retrieve and analyze all relevant features of the book nodes.(END OF REFLECTION)

        """,
"dblp": """Previous trial:
Question: How many papers have John Doe (Stanford University, Stanford, CA) and Jane Smith (Stanford University, Palo Alto, CA) co-authored?
Thought 1: Find the nodes for John Doe and Jane Smith.
Action 1: Retrieve[John Doe], Retrieve[Jane Smith]
Observation 1: The ID of John Doe's node is author-123456, and the ID of Jane Smith's node is author-789012.
Thought 2: Verify their organizations.
Action 2: Feature[author-123456, organization], Feature[author-789012, organization]
Observation 2: The organization of author-123456 is Stanford University, Stanford, CA. The organization of author-789012 is Stanford University, Palo Alto, CA.
Thought 3: Both are from Stanford University. Check co-authored papers using their paper neighbors.
Action 3: Degree[author-123456, paper] / Degree[author-789012, paper]
Observation 3: The paper neighbor node degree of author-123456 are: 1. The paper neighbor node degree of author-789012 are: 6. 
Thought 4: They have co-authored 6 papers.
Action 4: Finish[6]
Observation 4: Answer is 6

Reflection: 
- Recap of the Trial
  - Question: How many papers have John Doe (Stanford University, Stanford, CA) and Jane Smith (Stanford University, Palo Alto, CA) co-authored?
  - Graph Information Used:
    Retrieve[John Doe] and Retrieve[Jane Smith]: Located the nodes for the two authors.
    Feature[author-123456, organization] and Feature[author-789012, organization]: Confirmed they are from the same organization.
    Degree[author-123456, paper] and Degree[author-789012, paper]: Attempted to calculate the number of co-authored papers.
  - Outcome: Incorrectly guessed the answer using individual node degrees.
- Guided Reflection
  -  Understanding the Question
    - Core Goal: Identify the exact number of co-authored papers by finding shared paper neighbors.
    - Missed Information: Failed to check the Neighbor for both authors to identify common papers.
    - Misinterpretations: Assumed individual degrees equaled co-authored papers.
  - Analysis of Selected Graph Information
    - Relevance: Retrieve: Correctly identified the nodes for the two authors. Feature: Appropriately used to confirm the organization of the authors. Degree: Misused as a proxy for co-authored paper count.
    - Missed Insights: Did not retrieve and compare the actual paper neighbors to identify overlap. Overlooked the need to verify co-authorship explicitly through Neighbor.
    - Redundancies: Checking the node degree for individual authors without retrieving the actual neighbor lists was redundant.
  - Align the Problem Understanding and Graph Function Information
    - I failed to align the goal of finding shared paper nodes with the graph functions available for exploring relationships. I should use Neighbor to retrieve the paper neighbors for each author and compare the retrieved neighbor lists to find overlap, accurately reflecting the number of co-authored papers.
- Improved Strategy
  - Updated Understanding of the Problem:
    Determine co-authored papers by finding and counting shared paper neighbors.
  - Revised Graph Selection:
    Use Retrieve[John Doe] and Retrieve[Jane Smith] to locate the nodes for the authors.
    Use Feature to confirm their organizational affiliations if needed for context.
    Use Neighbor[author-123456, paper] and Neighbor[author-789012, paper] to retrieve their respective paper neighbors.
    Compare the lists of paper neighbors to identify common nodes (co-authored papers).
    Count the number of common papers and provide the result.
  - Avoiding Past Issues: 
    Avoid relying on Degree as a measure of shared connections without verification.
    Ensure neighbor relationships are explicitly retrieved and analyzed for overlap.(END OF REFLECTION)
        """,
        }


REFLECT_EXAMPLES_BASE={
"maple": """Previous trial Details:
Solve a question answering task with interleaving Thought, Interaction with Graph, Feedback from Graph steps. In Thought step, you can think about what further information is needed, and In Interaction step, you can get feedback from graphs with four functions: 
(1) Retrieve[keyword], which retrieves the related node from the graph according to the corresponding query.
(2) Feature[Node, feature], which returns the detailed attribute information of Node regarding the given "feature" key.
(3) Degree[Node, neighbor_type], which calculates the number of "neighbor_type" neighbors of the node Node in the graph.
(4) Neighbor[Node, neighbor_type], which lists the "neighbor_type" neighbours of the node Node in the graph and returns them.
Definition of the graph: There are three types of nodes in the graph: paper, author and venue.
Paper nodes have features: title, abstract, year and label. Author nodes have features: name. Venue nodes have features: name.
Paper nodes are linked to author nodes, venue nodes, reference nodes and cited_by nodes. Author nodes are linked to paper nodes. Venue nodes are linked to paper nodes.
Question: Who can be defined as the closest collaborator of author Alice B. Cooper, according to the number of their joint works? Please answer by providing node main feature (e.g., names) rather than node IDs.
Thought 1: The question is asking the closest collaborator of an author (Alice B. Cooper). We need to find the node in the graph.
Action 1: Retrieve[Alice B. Cooper]
Observation 1: The ID of this retrieval target node is 2676686768.
Thought 2: The question is asking the closest collaborator of an author, we need to calculate the "author" neighbor degree of this node to find the collaborators.
Action 2: Degree[2676686768, author]
Observation 2: The node or neighbor type does not exist in the graph. This might because your given neighbor type is not correct. Please modify it.
Thought 3: The question is asking the closest collaborator of an author, we need to check the node's author neighbor from the graph.
Action 3: Neighbor[2676686768, author]
Observation 3: The node or neighbor type does not exist in the graph. This might because your given neighbor type is not correct. Please modify it.
Thought 4: It seems that there is an issue with the graph structure or the given neighbor type.
Action 4: Finish[Unable to retrieve closest collaborator]
Observation 4: Answer is Unable to retrieve closest collaborator.

Reflection: I misunderstood the graph's node relationships, using the incorrect "author" neighbor type for an author node. I should first retrieve the author's "paper" neighbors and then identify the co-author with the most collaborations through those papers.
(END OF REFLECTION)
            """,
"amazon": """Previous Trial:
Solve a question answering task with interleaving Thought, Interaction with Graph, Feedback from Graph steps. In Thought step, you can think about what further information is needed, and In Interaction step, you can get feedback from graphs with four functions: 
(1) Retrieve[keyword], which retrieves the related node from the graph according to the corresponding query.
(2) Feature[Node, feature], which returns the detailed attribute information of Node regarding the given "feature" key.
(3) Degree[Node, neighbor_type], which calculates the number of "neighbor_type" neighbors of the node Node in the graph.
(4) Neighbor[Node, neighbor_type], which lists the "neighbor_type" neighbours of the node Node in the graph and returns them.
Definition of the graph: There are two types of nodes in the graph: item and brand.
Item nodes have features: title, description, price, img, category. Brand nodes have features: name.
Item nodes are linked to their brand nodes, also_viewed_item nodes, buy_after_viewing_item nodes, also_bought_item nodes, bought_together_item nodes. Brand nodes are linked to their item nodes.Please answer by providing node main feature (e.g., names) rather than node IDs.
Question: What is the count of co-viewed items for item Columbia Women's Silver Ridge Lite Long Sleeve Shirt? Please answer by providing node main feature (e.g., names) rather than node IDs.
Thought 1: The question is asking the count of co-viewed items for a item node (Columbia Women's Silver Ridge Lite Long Sleeve Shirt). We need to find the node in the graph.
Action 1: Retrieve[Columbia Women's Silver Ridge Lite Long Sleeve Shirt]
Observation 1: The ID of this retrieval target node is ABCDE00000.
Thought 2: The question is asking the count of co-viewed items for an item, we need to calculate the node's also_viewed_item neighbor degree from the graph.
Action 2: Degree[ABCDE00000, also_viewed_item]
Observation 2: The also_viewed_item neighbor node degree of ABCDE00000 are: 15. 
Thought 3: The question is asking the count of co-viewed items for an item, we need to list the node's also_viewed_item neighbors from the graph.
Action 3: Neighbor[ABCDE00000, also_viewed_item]
Observation 3: The also_viewed_item neighbors of ABCDE00000 are: ['B009XNJPX8', 'B00FAN8CF2', 'B0043EVJJA', 'B009XNJOBQ', 'B0043EVIVE'].
Thought 4:The co-viewed items for item Columbia Women's Silver Ridge Lite Long Sleeve Shirt are 'B009XNJPX8', 'B00FAN8CF2', 'B0043EVJJA', 'B009XNJOBQ', 'B0043EVIVE'.
Action 4:Finish['B009XNJPX8', 'B00FAN8CF2', 'B0043EVJJA', 'B009XNJOBQ', 'B0043EVIVE']
Observation 4: Answer is 'B009XNJPX8', 'B00FAN8CF2', 'B0043EVJJA', 'B009XNJOBQ', 'B0043EVIVE'

Reflection: I listed the ID of the node's also_viewed_item neighbor as the answer while the question only asked the count of co-viewed items, I should return the also_viewed_item neighbor node degree as the answer before.
(END OF REFLECTION)
            """,
"biomedical": """Previous Trial:
Solve a question answering task with interleaving Thought, Interaction with Graph, Feedback from Graph steps. In Thought step, you can think about what further information is needed, and In Interaction step, you can get feedback from graphs with four functions: 
(1) Retrieve[keyword], which retrieves the related node from the graph according to the corresponding query.
(2) Feature[Node, feature], which returns the detailed attribute information of Node regarding the given "feature" key.
(3) Degree[Node, neighbor_type], which calculates the number of "neighbor_type" neighbors of the node Node in the graph.
(4) Neighbor[Node, neighbor_type], which lists the "neighbor_type" neighbours of the node Node in the graph and returns them.
Definition of the graph: There are eleven types of nodes in the graph: Anatomy, Biological Process, Cellular Component, Compound, Disease, Gene, Molecular Function, Pathway, Pharmacologic Class, Side Effect, Symptom.
Each node has name feature.
There are these types of edges: Anatomy-downregulates-Gene, Anatomy-expresses-Gene, Anatomy-upregulates-Gene, Compound-binds-Gene, Compound-causes-Side Effect, Compound-downregulates-Gene, Compound-palliates-Disease, Compound-resembles-Compound, Compound-treats-Disease, Compound-upregulates-Gene, Disease-associates-Gene, Disease-downregulates-Gene, Disease-localizes-Anatomy, Disease-presents-Symptom, Disease-resembles-Disease, Disease-upregulates-Gene, Gene-covaries-Gene, Gene-interacts-Gene, Gene-participates-Biological Process, Gene-participates-Cellular Component, Gene-participates-Molecular Function, Gene-participates-Pathway, Gene-regulates-Gene, Pharmacologic Class-includes-Compound.
Question: Could you tell me the side effects of the compound Aspirin? Please answer by providing node main feature (e.g., names) rather than node IDs.
Thought 1: The question is related to a compound node (Aspirin). We need to find the node in the graph.
Action 1: Retrieve[Aspirin]
Observation 1: The ID of this retrieval target node is DB00945.
Thought 2: The question is asking the side effects of the compound. We need to check the node's 'Compound-causes-Side Effect' neighbor from the graph.
Action 2: Neighbor[DB00945, Compound-causes-Side Effect]
Observation 2: The Compound-causes-Side Effect neighbors of DB00945 are: ['C1234567'].
Thought 3: The side effect of Aspirin is C1234567.
Action 3: Finish['C1234567']
Observation 3: Answer is 'C1234567'

Reflection: I only provided the ID of the side effect of Aspirin as the answer, I should provide node name feature rather than node IDs.
(END OF REFLECTION)
            """,
"legal": """Previous Trial:
Solve a question answering task with interleaving Thought, Interaction with Graph, Feedback from Graph steps. In Thought step, you can think about what further information is needed, and In Interaction step, you can get feedback from graphs with four functions: 
(1) Retrieve[keyword], which retrieves the related node from the graph according to the corresponding query.
(2) Feature[Node, feature], which returns the detailed attribute information of Node regarding the given "feature" key.
(3) Degree[Node, neighbor_type], which calculates the number of "neighbor_type" neighbors of the node Node in the graph.
(4) Neighbor[Node, neighbor_type], which lists the "neighbor_type" neighbours of the node Node in the graph and returns them.
Definition of the graph: There are four types of nodes in the graph: opinion, opinion_cluster, docket, and court.
Opinion nodes have features: plain_text. Opinion_cluster nodes have features: syllabus, judges, case_name, attorneys. Docket nodes have features: pacer_case_id, case_name. Court nodes have features: full_name, start_date, end_date, citation_string.
Opinion nodes are linked to their reference nodes and cited_by nodes, as well as their opinion_cluster nodes. Opinion_cluster nodes are linked to opinion nodes and docket nodes. Docket nodes are linked to opinion_cluster nodes and court nodes. Court nodes are linked to docket nodes.
Question: which court is handling the case listed under the PACER docket number 108735.0? Please answer by providing node main feature (e.g., names) rather than node IDs.
Thought 1: The question is asking for the court handling the case represented by the docket node with the given PACER case ID. We need to find the docket node first.
Action 1: Retrieve[108735.0, Docket]
Observation 1: The ID of this retrieval target node is d-65685551.
Thought 2: Now that we have the docket node, we need to find the associated court node by checking its 'court' neighbor.
Action 2: Neighbor[d-65685551, court]
Observation 2: The court neighbors of d-65685551 are: ['ca9']. 
Thought 3: The court that is handling the case is 'ca9'.
Action 3: Finish[ca9]
Observation 3: Answer is ca9

Reflection: I only provided the ID of the court that is handling the case, I should provide node name feature rather than node IDs.
(END OF REFLECTION)
        """,
"goodreads": """Previous Trial:
Solve a question answering task with interleaving Thought, Interaction with Graph, Feedback from Graph steps. In Thought step, you can think about what further information is needed, and In Interaction step, you can get feedback from graphs with four functions: 
(1) Retrieve[keyword], which retrieves the related node from the graph according to the corresponding query.
(2) Feature[Node, feature], which returns the detailed attribute information of Node regarding the given "feature" key.
(3) Degree[Node, neighbor_type], which calculates the number of "neighbor_type" neighbors of the node Node in the graph.
(4) Neighbor[Node, neighbor_type], which lists the "neighbor_type" neighbours of the node Node in the graph and returns them.
Definition of the graph: There are four types of nodes in the graph: book, author, publisher, and series. 
Book nodes have features: country_code, language_code, is_ebook, title, description, format, num_pages, publication_year, url, popular_shelves, and genres. Author nodes have features: name. Publisher nodes have features: name. Series nodes have features: title and description. 
Book nodes are linked to their author nodes, publisher nodes, series nodes and similar_books nodes. Author nodes are linked to their book nodes. Publisher nodes are linked to their book nodes. Series nodes are linked to their book nodes.
Question: What is the most common publication format of books by author Jane Austen? Please answer by providing node main feature (e.g., names) rather than node IDs.
Thought 1: We need to find the author node for Jane Austen to determine the most common publication format of his books.
Action 1: Retrieve[Jane Austen]
Observation 1: The ID of this retrieval target node is author-105.
Thought 2: Now that we have the author node, we need to calculate the node's book neighbor degree from the graph.
Action 2: Degree[author-105, book]
Observation 2: The book neighbor node degree of author-281126 are: 1. 
Thought 3: The number of books is 1.
Action 3: Finish[paperback]
Observation 3: Answer is paperback

Reflection: I misunderstood the number of books (Degree) directly determined the most common format and I should verify and analyze node features.
(END OF REFLECTION)
        """,
"dblp": """Previous Trial:
Solve a question answering task with interleaving Thought, Interaction with Graph, Feedback from Graph steps. In Thought step, you can think about what further information is needed, and In Interaction step, you can get feedback from graphs with four functions: 
(1) Retrieve[keyword], which retrieves the related node from the graph according to the corresponding query.
(2) Feature[Node, feature], which returns the detailed attribute information of Node regarding the given "feature" key.
(3) Degree[Node, neighbor_type], which calculates the number of "neighbor_type" neighbors of the node Node in the graph.
(4) Neighbor[Node, neighbor_type], which lists the "neighbor_type" neighbours of the node Node in the graph and returns them.
Definition of the graph: There are three types of nodes in the graph: paper, author and venue.
Paper nodes have features: title, abstract, keywords, lang, and year. Author nodes have features: name and organization. Venue nodes have features: name.
Paper nodes are linked to their author nodes, venue nodes, reference nodes (the papers this paper cite) and cited_by nodes (other papers which cite this paper). Author nodes are linked to their paper nodes. Venue nodes are linked to their paper nodes.
Question: What is the number of papers that John Doe from Stanford University, Stanford, CA and Jane Smith from Stanford University, Palo Alto, CA co-authored? Please answer by providing node main feature (e.g., names) rather than node IDs.
Thought 1: We need to find the nodes of these two researchers in the graph to determine their co-authored papers
Action 1: Retrieve[John Doe], Retrieve[Jane Smith]
Observation 1: The ID of John Doe's node is author-123456, and the ID of Jane Smith's node is author-789012.
Thought 2: To confirm they are from the same organization, we need to check the organization feature of their nodes.
Action 2: Feature[author-123456, organization], Feature[author-789012, organization]
Observation 2: The organization of author-123456 is Stanford University, Stanford, CA. The organization of author-789012 is Stanford University, Palo Alto, CA.
Thought 3: It's the same organization, now we can calculate the number of papers they co-authored.
Action 3: Degree[author-123456, paper] / Degree[author-789012, paper]
Observation 3: The paper neighbor node degree of author-123456 are: 1. The paper neighbor node degree of author-789012 are: 6. 
Thought 4: There are six papers that they all write.
Action 4: Finish[6]
Observation 4: Answer is 6

Reflection: I misunderstood that node degrees represent shared connections and I should compare neighbor lists to identify shared connections rather than relying solely on node degrees.
(END OF REFLECTION)
        """,
        }

REFLECT_EXAMPLES_PLAN_BASE={
"maple": """Previous trial Details:
Solve a question answering task with interleaving Thought, Interaction with Graph, Feedback from Graph steps. In Thought step, you can think about what further information is needed, and In Interaction step, you can get feedback from graphs with four functions: 
(1) Retrieve[keyword], which retrieves the related node from the graph according to the corresponding query.
(2) Feature[Node, feature], which returns the detailed attribute information of Node regarding the given "feature" key.
(3) Degree[Node, neighbor_type], which calculates the number of "neighbor_type" neighbors of the node Node in the graph.
(4) Neighbor[Node, neighbor_type], which lists the "neighbor_type" neighbours of the node Node in the graph and returns them.
Definition of the graph: There are three types of nodes in the graph: paper, author and venue.
Paper nodes have features: title, abstract, year and label. Author nodes have features: name. Venue nodes have features: name.
Paper nodes are linked to author nodes, venue nodes, reference nodes and cited_by nodes. Author nodes are linked to paper nodes. Venue nodes are linked to paper nodes.
Question: Who can be defined as the closest collaborator of author Alice B. Cooper, according to the number of their joint works? Please answer by providing node main feature (e.g., names) rather than node IDs.
plan 1: The question requires identifying the closest collaborator of "Alice B. Cooper" based on the number of joint works. Currently, we need to locate the node for Alice B. Cooper in the graph. 
Thought 1: We need to find the node for Alice B. Cooper in the graph.
Action 1: Retrieve[Alice B. Cooper]
Observation 1: The ID of this retrieval target node is 2676686768.
plan 2: The question requires determine the closest collaborator. We have the node ID of Alice B. Cooper. Thus, we need to count the number of collaborators.
Thought 2: We need to calculate the neighbor degree of this node to identify collaborators.
Action 2: Degree[2676686768, author]
Observation 2: The node or neighbor type does not exist in the graph. This might because your given neighbor type is not correct. Please modify it.
plan 3: The question is asking the closest collaborator of an author. We have the node ID of Alice B. Cooper. Since calculating the degree failed, we should explicitly check the author neighbors of this node and get their name.
Thought 3: We need to check the node's author neighbor from the graph.
Action 3: Feature[Neighbor[2676686768, author], name]
Observation 3: The node or neighbor type does not exist in the graph. This might because your given neighbor type is not correct. Please modify it.
plan 4: The question is asking the closest collaborator of an author, we are unable to retrieve the closest collaborator.
Thought 4: It seems that there is an issue with the graph structure or the given neighbor type, due to issues with the neighbor type.
Action 4: Finish[Unable to retrieve closest collaborator]
Observation 4: Answer is Unable to retrieve closest collaborator

Reflection: I misunderstood the graph's node relationships, using the incorrect "author" neighbor type for an author node. I should first retrieve the author's "paper" neighbors and then identify the co-author with the most collaborations through those papers.
(END OF REFLECTION)
            """,
"amazon": """Previous Trial:
Solve a question answering task with interleaving Thought, Interaction with Graph, Feedback from Graph steps. In Thought step, you can think about what further information is needed, and In Interaction step, you can get feedback from graphs with four functions: 
(1) Retrieve[keyword], which retrieves the related node from the graph according to the corresponding query.
(2) Feature[Node, feature], which returns the detailed attribute information of Node regarding the given "feature" key.
(3) Degree[Node, neighbor_type], which calculates the number of "neighbor_type" neighbors of the node Node in the graph.
(4) Neighbor[Node, neighbor_type], which lists the "neighbor_type" neighbours of the node Node in the graph and returns them.
Definition of the graph: There are two types of nodes in the graph: item and brand.
Item nodes have features: title, description, price, img, category. Brand nodes have features: name.
Item nodes are linked to their brand nodes, also_viewed_item nodes, buy_after_viewing_item nodes, also_bought_item nodes, bought_together_item nodes. Brand nodes are linked to their item nodes.Please answer by providing node main feature (e.g., names) rather than node IDs.
Question: What is the count of co-viewed items for item Columbia Women's Silver Ridge Lite Long Sleeve Shirt? Please answer by providing node main feature (e.g., names) rather than node IDs.
Plan 1: The question is asking the count of co-viewed items for a item node (Columbia Women's Silver Ridge Lite Long Sleeve Shirt), we need to find the node first.
Thought 1: We need to find the node in the graph.
Action 1: Retrieve[Columbia Women's Silver Ridge Lite Long Sleeve Shirt]
Observation 1: The ID of this retrieval target node is ABCDE00000.
Plan 2: The question is asking the count of co-viewed items for an item. We have the node ID of Columbia Women's Silver Ridge Lite Long Sleeve Shirt. Thus, we need to calculate the count of node's also_viewed_item neighbors.
Thought 2: We need to calculate the node's also_viewed_item neighbor degree from the graph.
Action 2: Degree[ABCDE00000, also_viewed_item]
Observation 2: The also_viewed_item neighbor node degree of ABCDE00000 are: 15. 
Plan3:  The question is asking the count of co-viewed items for an item. We have the node ID of the item and its co-view count, so we need to list the node's also_viewed_item neighbors to identify the co-viewed items by their features.
Thought 3: We need to list the node's also_viewed_item neighbors from the graph.
Action 3: Neighbor[ABCDE00000, also_viewed_item]
Observation 3: The also_viewed_item neighbors of ABCDE00000 are: ['B009XNJPX8', 'B00FAN8CF2', 'B0043EVJJA', 'B009XNJOBQ', 'B0043EVIVE'].
Plan 4: The question is asking for the count of co-viewed items. We have the co-viewed items listed, so we need to provide them.
Thought 4:The co-viewed items for item Columbia Women's Silver Ridge Lite Long Sleeve Shirt are 'B009XNJPX8', 'B00FAN8CF2', 'B0043EVJJA', 'B009XNJOBQ', 'B0043EVIVE'.
Action 4:Finish['B009XNJPX8', 'B00FAN8CF2', 'B0043EVJJA', 'B009XNJOBQ', 'B0043EVIVE']
Observation 4: Answer is 'B009XNJPX8', 'B00FAN8CF2', 'B0043EVJJA', 'B009XNJOBQ', 'B0043EVIVE'

Reflection: I listed the ID of the node's also_viewed_item neighbor as the answer while the question only asked the count of co-viewed items, I should return the also_viewed_item neighbor node degree as the answer before.
(END OF REFLECTION)
            """,
"biomedical": """Previous Trial:
Solve a question answering task with interleaving Thought, Interaction with Graph, Feedback from Graph steps. In Thought step, you can think about what further information is needed, and In Interaction step, you can get feedback from graphs with four functions: 
(1) Retrieve[keyword], which retrieves the related node from the graph according to the corresponding query.
(2) Feature[Node, feature], which returns the detailed attribute information of Node regarding the given "feature" key.
(3) Degree[Node, neighbor_type], which calculates the number of "neighbor_type" neighbors of the node Node in the graph.
(4) Neighbor[Node, neighbor_type], which lists the "neighbor_type" neighbours of the node Node in the graph and returns them.
Definition of the graph: There are eleven types of nodes in the graph: Anatomy, Biological Process, Cellular Component, Compound, Disease, Gene, Molecular Function, Pathway, Pharmacologic Class, Side Effect, Symptom.
Each node has name feature.
There are these types of edges: Anatomy-downregulates-Gene, Anatomy-expresses-Gene, Anatomy-upregulates-Gene, Compound-binds-Gene, Compound-causes-Side Effect, Compound-downregulates-Gene, Compound-palliates-Disease, Compound-resembles-Compound, Compound-treats-Disease, Compound-upregulates-Gene, Disease-associates-Gene, Disease-downregulates-Gene, Disease-localizes-Anatomy, Disease-presents-Symptom, Disease-resembles-Disease, Disease-upregulates-Gene, Gene-covaries-Gene, Gene-interacts-Gene, Gene-participates-Biological Process, Gene-participates-Cellular Component, Gene-participates-Molecular Function, Gene-participates-Pathway, Gene-regulates-Gene, Pharmacologic Class-includes-Compound.
Question: Could you tell me the side effects of the compound Aspirin? Please answer by providing node main feature (e.g., names) rather than node IDs.
Plan 1: The question is asking for the side effects of the compound Aspirin. We need to locate the node for Aspirin.
Thought 1: We need to find the node in the graph.
Action 1: Retrieve[Aspirin]
Observation 1: The ID of this retrieval target node is DB00945.
Plan 2: The question is asking for the side effects of the compound Aspirin. We have the node ID of Aspirin as DB00945. Thus, we need to check the node's Compound-causes-Side Effect neighbors in the graph.
Thought 2: We need to check the node's 'Compound-causes-Side Effect' neighbor from the graph.
Action 2: Neighbor[DB00945, Compound-causes-Side Effect]
Observation 2: The Compound-causes-Side Effect neighbors of DB00945 are: ['C1234567'].
Plan 3: The question is asking for the side effects of the compound Aspirin. We have the Compound-causes-Side Effect neighbor. Thus, we need to identify the side effects and return them as the answer.
Thought 3: The side effect of Aspirin is C1234567.
Action 3: Finish['C1234567']

Reflection: I only provided the ID of the side effect of Aspirin as the answer, I should provide node name feature rather than node IDs.
(END OF REFLECTION)
            """,
"legal": """Previous Trial:
Solve a question answering task with interleaving Thought, Interaction with Graph, Feedback from Graph steps. In Thought step, you can think about what further information is needed, and In Interaction step, you can get feedback from graphs with four functions: 
(1) Retrieve[keyword], which retrieves the related node from the graph according to the corresponding query.
(2) Feature[Node, feature], which returns the detailed attribute information of Node regarding the given "feature" key.
(3) Degree[Node, neighbor_type], which calculates the number of "neighbor_type" neighbors of the node Node in the graph.
(4) Neighbor[Node, neighbor_type], which lists the "neighbor_type" neighbours of the node Node in the graph and returns them.
Definition of the graph: There are four types of nodes in the graph: opinion, opinion_cluster, docket, and court.
Opinion nodes have features: plain_text. Opinion_cluster nodes have features: syllabus, judges, case_name, attorneys. Docket nodes have features: pacer_case_id, case_name. Court nodes have features: full_name, start_date, end_date, citation_string.
Opinion nodes are linked to their reference nodes and cited_by nodes, as well as their opinion_cluster nodes. Opinion_cluster nodes are linked to opinion nodes and docket nodes. Docket nodes are linked to opinion_cluster nodes and court nodes. Court nodes are linked to docket nodes.
Question: which court is handling the case listed under the PACER docket number 108735.0? Please answer by providing node main feature (e.g., names) rather than node IDs.
Plan1: The question is asking for the court handling the case listed under the PACER docket number 108735.0. We need to find the corresponding docket node in the graph. 
Thought 1: We need to find the docket node first.
Action 1: Retrieve[108735.0, Docket]
Observation 1: The ID of this retrieval target node is d-65685551.
Plan2: The question is asking for the court handling the case represented by the docket node. We have the docket node ID as d-65685551. Thus, we need to check its court neighbor in the graph to identify the associated court.
Thought 2: We need to find the associated court node by checking its 'court' neighbor.
Action 2: Neighbor[d-65685551, court]
Observation 2: The court neighbors of d-65685551 are: ['ca9']. 
Plan3: The question is asking for the court handling the case. We have identified the court neighbor. Thus, we need to provide the name of the court as the answer.
Thought 3: The court that is handling the case is 'ca9'.
Action 3: Finish[ca9]
Observation 3: Answer is ca9

Reflection: I only provided the ID of the court that is handling the case, I should provide node name feature rather than node IDs.
(END OF REFLECTION)
        """,
"goodreads": """Previous Trial:
Solve a question answering task with interleaving Thought, Interaction with Graph, Feedback from Graph steps. In Thought step, you can think about what further information is needed, and In Interaction step, you can get feedback from graphs with four functions: 
(1) Retrieve[keyword], which retrieves the related node from the graph according to the corresponding query.
(2) Feature[Node, feature], which returns the detailed attribute information of Node regarding the given "feature" key.
(3) Degree[Node, neighbor_type], which calculates the number of "neighbor_type" neighbors of the node Node in the graph.
(4) Neighbor[Node, neighbor_type], which lists the "neighbor_type" neighbours of the node Node in the graph and returns them.
Definition of the graph: There are four types of nodes in the graph: book, author, publisher, and series. 
Book nodes have features: country_code, language_code, is_ebook, title, description, format, num_pages, publication_year, url, popular_shelves, and genres. Author nodes have features: name. Publisher nodes have features: name. Series nodes have features: title and description. 
Book nodes are linked to their author nodes, publisher nodes, series nodes and similar_books nodes. Author nodes are linked to their book nodes. Publisher nodes are linked to their book nodes. Series nodes are linked to their book nodes.
Question: What is the most common publication format of books by author Jane Austen? Please answer by providing node main feature (e.g., names) rather than node IDs.
Plan 1: The question is asking for the most common publication format of books by the author Jane Austen. We have the author's name. Thus, we need to find the corresponding author node in the graph.
Thought 1: We need to find the author node for Jane Austen.
Action 1: Retrieve[Jane Austen]
Observation 1: The ID of this retrieval target node is author-105.
Plan 2: The question is asking for the publication format of books written by Jane Austen. We have the author node ID as author-105. Thus, we need to calculate the degree of its book neighbors to determine how many books are connected to this author.
Thought 2: We need to calculate the node's book neighbor degree from the graph.
Action 2: Degree[author-105, book]
Observation 2: The book neighbor node degree of author-281126 are: 1. 
Plan 3: The question is asking for the most common publication format of books by Jane Austen. We have determined there is only 1 book, so we need to identify its publication format and provide it as the answer.
Thought 3: The number of books is 1.
Action 3: Finish[paperback]
Observation 3: Answer is paperback

Reflection: I misunderstood the number of books (Degree) directly determined the most common format and I should verify and analyze node features.
(END OF REFLECTION)
        """,
"dblp": """Previous Trial:
Solve a question answering task with interleaving Thought, Interaction with Graph, Feedback from Graph steps. In Thought step, you can think about what further information is needed, and In Interaction step, you can get feedback from graphs with four functions: 
(1) Retrieve[keyword], which retrieves the related node from the graph according to the corresponding query.
(2) Feature[Node, feature], which returns the detailed attribute information of Node regarding the given "feature" key.
(3) Degree[Node, neighbor_type], which calculates the number of "neighbor_type" neighbors of the node Node in the graph.
(4) Neighbor[Node, neighbor_type], which lists the "neighbor_type" neighbours of the node Node in the graph and returns them.
Definition of the graph: There are three types of nodes in the graph: paper, author and venue.
Paper nodes have features: title, abstract, keywords, lang, and year. Author nodes have features: name and organization. Venue nodes have features: name.
Paper nodes are linked to their author nodes, venue nodes, reference nodes (the papers this paper cite) and cited_by nodes (other papers which cite this paper). Author nodes are linked to their paper nodes. Venue nodes are linked to their paper nodes.
Question: What is the number of papers that John Doe from Stanford University, Stanford, CA and Jane Smith from Stanford University, Palo Alto, CA co-authored? Please answer by providing node main feature (e.g., names) rather than node IDs.
Plan 1: The question is asking for the number of papers co-authored by John Doe from Stanford University, Stanford, CA, and Jane Smith from Stanford University, Palo Alto, CA. We need to find the corresponding nodes in the graph for each. 
Thought 1: We need to find the nodes of these two researchers in the graph.
Action 1: Retrieve[John Doe], Retrieve[Jane Smith]
Observation 1: The ID of John Doe's node is author-123456, and the ID of Jane Smith's node is author-789012.
Plan 2: The question is asking to confirm whether both researchers are from the same organization. We have their node IDs. Thus, we need to check the organization feature of their respective nodes.
Thought 2: We need to check the organization feature of their nodes.
Action 2: Feature[author-123456, organization], Feature[author-789012, organization]
Observation 2: The organization of author-123456 is Stanford University, Stanford, CA. The organization of author-789012 is Stanford University, Palo Alto, CA.
Plan 3: The question is asking for the number of papers co-authored by these researchers. We have confirmed they are from the same organization. Thus, we need to calculate the number of papers both have authored and identify the overlap (co-authored papers).
Thought 3: We can calculate the number of papers they co-authored.
Action 3: Degree[author-123456, paper] / Degree[author-789012, paper]
Observation 3: The paper neighbor node degree of author-123456 are: 1. The paper neighbor node degree of author-789012 are: 6. 
Plan 4: The question is asking for the number of papers co-authored by the two researchers. We have determined their individual paper counts. Thus, we need to identify the overlap (co-authored papers) and provide the final count.
Thought 4: There are six papers that they all write.
Action 4: Finish[6]
Observation 4: Answer is 6

Reflection: I misunderstood that node degrees represent shared connections and I should compare neighbor lists to identify shared connections rather than relying solely on node degrees.
(END OF REFLECTION)
        """,
        }

EVAL_EXAMPLES={
    "maple": """Solve a question answering task with interleaving Thought, Interaction with Graph, Feedback from Graph steps. In Thought step, you can think about what further information is needed, and In Interaction step, you can get feedback from graphs with four functions: 
        (1) Retrieve[keyword], which retrieves the related node from the graph according to the corresponding query.
        (2) Feature[Node, feature], which returns the detailed attribute information of Node regarding the given "feature" key.
        (3) Degree[Node, neighbor_type], which calculates the number of "neighbor_type" neighbors of the node Node in the graph.
        (4) Neighbor[Node, neighbor_type], which lists the "neighbor_type" neighbours of the node Node in the graph and returns them.
        Definition of the graph: There are three types of nodes in the graph: paper, author and venue.
        Paper nodes have features: title, abstract, year and label. Author nodes have features: name. Venue nodes have features: name.
        Paper nodes are linked to their author nodes, venue nodes, reference paper nodes and cited_by paper nodes. Author nodes are linked to their paper nodes. Venue nodes are linked to their paper nodes.
        Question: Whom has the author Emily T. Anderson often been working with? Please answer by providing node main feature (e.g., names) rather than node IDs.
        Thought 1: The question asks for co-authors of Emily T. Anderson. We need to locate the node representing Emily T. Anderson in the graph.
        Action 1: Retrieve[Emily T. Anderson]
        Observation 1: The ID of this retrieval target node is 3456789123.
        Thought 2: To find co-authors, we need to identify the papers authored by Emily T. Anderson.
        Action 2: Neighbor[3456789123, paper]
        Observation 2: The paper neighbors of 3456789123 are: ['4012345678', '4098765432']. 
        Thought 3: We have identified the papers authored by Emily T. Anderson. To find her co-authors, we need to retrieve the authors of these papers.
        Action 3: Neighbor[4012345678, author], Neighbor[4098765432, author]
        Observation 3: The author neighbors of paper 4012345678 are: ['3456789123', '2233445566', '3344556677'].The author neighbors of paper 4098765432 are: ['3456789123', '5566778899', '2233445577'].
        Thought 4: We now have a list of co-authors. Next, we need to retrieve their names.
        Action 4: Feature[3456789123, name], Feature[2233445566, name], Feature[3344556677, name], Feature[3456789123, name], Feature[5566778899, name], Feature[2233445577, name]
        Observation 4: The name feature of 3456789123 are: John D. Miller. The name feature of 2233445566 are: Sarah K. Williams. The name feature of 3344556677 are: Mark A. Johnson. The name feature of 3456789123 are: John D. Miller. The name feature of 5566778899 are: Anna M. Green. The name feature of 2233445577 are: David P. Brown. 
        Thought 5: The co-authors of Emily T. Anderson are John D. Miller, Sarah K. Williams, Mark A. Johnson,  Anna M. Green, and David P. Brown.
        Action 5: Finish[John D. Miller, Sarah K. Williams, Mark A. Johnson,  Anna M. Green, and David P. Brown]
        Observation 5: Answer is John D. Miller, Sarah K. Williams, Mark A. Johnson,  Anna M. Green, and David P. Brown

        Proceed with explanation and judgment below:
        Explanation: The process identified the papers authored by Emily T. Anderson and retrieved the associated co-authors correctly. However, there was a logical misstep in Thought 4, where the question required identifying the co-authors Emily T. Anderson often works with. This implies we need to determine the most frequent collaborators.
        Judgment: [No]

        Solve a question answering task with interleaving Thought, Interaction with Graph, Feedback from Graph steps. In Thought step, you can think about what further information is needed, and In Interaction step, you can get feedback from graphs with four functions: 
        (1) Retrieve[keyword], which retrieves the related node from the graph according to the corresponding query.
        (2) Feature[Node, feature], which returns the detailed attribute information of Node regarding the given "feature" key.
        (3) Degree[Node, neighbor_type], which calculates the number of "neighbor_type" neighbors of the node Node in the graph.
        (4) Neighbor[Node, neighbor_type], which lists the "neighbor_type" neighbours of the node Node in the graph and returns them.
        Definition of the graph: There are three types of nodes in the graph: paper, author and venue.
        Paper nodes have features: title, abstract, year and label. Author nodes have features: name. Venue nodes have features: name.
        Paper nodes are linked to their author nodes, venue nodes, reference paper nodes and cited_by paper nodes. Author nodes are linked to their paper nodes. Venue nodes are linked to their paper nodes.
        Question: Where is the paper "nickel catalyzed denitrogenative annulation reactions of 1 2 3 benzotriazin 4 3h ones with 1 3 dienes and alkenes" published? Please answer by providing node main feature (e.g., names) rather than node IDs. 
        Thought 1: The question is asking information of a node (nickel catalyzed denitrogenative annulation reactions of 1 2 3 benzotriazin 4 3h ones with 1 3 dienes and alkenes). We need to find the node in the graph.
        Action 1: Retrieve[nickel catalyzed denitrogenative annulation reactions of 1 2 3 benzotriazin 4 3h ones with 1 3 dienes and alkenes]
        Observation 1: The ID of this retrieval target node is 1967836612.
        Thought 2: The question is asking the published venue of a paper, we need to check the node's venue neighbor from the graph.
        Action 2: Neighbor[1967836612, venue]
        Observation 2: The venue neighbors of 1967836612 are: ['205050996']. 
        Thought 3: The ID of the published venue is 205050996. We need to get its name.
        Action 3: Feature[205050996, name]
        Observation 3: The name feature of 205050996 are: journal of organic chemistry. 
        Thought 4: The name of the published venue is journal of organic chemistry.
        Action 4: Finish[journal of organic chemistry]
        Observation 4: Answer is journal of organic chemistry.

        Proceed with explanation and judgment below:
        Explanation: The final answer, "journal of organic chemistry," is directly derived from the graph data. The reasoning process accurately identifies the relevant "paper" node, retrieves its "venue" neighbor, and fetches the "name" feature of the venue node. Each step logically follows from the observations, and the answer directly satisfies the question without introducing irrelevant or incorrect elements.
        Judgment: [Yes]
        """,
    "amazon": """Solve a question answering task with interleaving Thought, Interaction with Graph, Feedback from Graph steps. In Thought step, you can think about what further information is needed, and In Interaction step, you can get feedback from graphs with four functions: 
        (1) Retrieve[keyword], which retrieves the related node from the graph according to the corresponding query.
        (2) Feature[Node, feature], which returns the detailed attribute information of Node regarding the given "feature" key.
        (3) Degree[Node, neighbor_type], which calculates the number of "neighbor_type" neighbors of the node Node in the graph.
        (4) Neighbor[Node, neighbor_type], which lists the "neighbor_type" neighbours of the node Node in the graph and returns them.
        Definition of the graph: There are two types of nodes in the graph: item and brand.
        Item nodes have features: title, description, price, img, category. Brand nodes have features: name.
        Item nodes are linked to their brand nodes, also_viewed_item nodes, buy_after_viewing_item nodes, also_bought_item nodes, bought_together_item nodes. Brand nodes are linked to their item nodes.Please answer by providing node main feature (e.g., names) rather than node IDs.
        Question: Which item shares over 4 co-viewed items with the item Healthy Home Cooking: Nutritious Recipes for Everyday Meals? Please answer by providing node main feature (e.g., names) rather than node IDs.
        Thought 1: To find an item that shares more than 4 co-viewed items with Healthy Home Cooking: Nutritious Recipes for Everyday Meals, I first need to find the ID of the item.
        Action 1: Retrieve[Healthy Home Cooking: Nutritious Recipes for Everyday Meals]
        Observation 1: The ID of this retrieval target node is 123456789.
        Thought 2: Now that I have the ID of the item "Healthy Home Cooking: Nutritious Recipes for Everyday Meals", I need to find its also_viewed_item neighbors.
        Action 2: Neighbor[123456789, also_viewed_item]
        Observation 2: The also_viewed_item neighbors of 123456789 are: ['158040250X', '0807847909', '0688162193', '0806520280', '0688100120', '0738212288', '146960762X', '0809231530', '0767903145', '0671796291', '0802132839', '1580087981', '0060187166', '1607745313', '0399146563', '161243097X', '1580402275', '080785655X', '159240085X', '1439193304', '1580400752', '0764569112', '1469608707', '0806526777']. 
        Thought 3: Next, I need to determine if any of these also_viewed_item neighbors share more than 4 co-viewed items with the original item.
        Action 3: Degree[158040250X, also_viewed_item]
        Observation 3: The also_viewed_item neighbor node degree of 158040250X are: 5. 
        Thought 4: The node degree of 158040250X is 5, which means it shares more than 4 co-viewed items with the original item.
        Action 4: Finish[158040250X]
        Observation 4: Answer is 158040250X

        Proceed with explanation and judgment below:
        Explanation: The answer assumes that the node degree represents shared co-viewed items without verifying overlap through a comparison of neighbor lists.
        Judgment: [No]

        Solve a question answering task with interleaving Thought, Interaction with Graph, Feedback from Graph steps. In Thought step, you can think about what further information is needed, and In Interaction step, you can get feedback from graphs with four functions: 
        (1) Retrieve[keyword], which retrieves the related node from the graph according to the corresponding query.
        (2) Feature[Node, feature], which returns the detailed attribute information of Node regarding the given "feature" key.
        (3) Degree[Node, neighbor_type], which calculates the number of "neighbor_type" neighbors of the node Node in the graph.
        (4) Neighbor[Node, neighbor_type], which lists the "neighbor_type" neighbours of the node Node in the graph and returns them.
        Definition of the graph: There are two types of nodes in the graph: item and brand.
        Item nodes have features: title, description, price, img, category. Brand nodes have features: name.
        Item nodes are linked to their brand nodes, also_viewed_item nodes, buy_after_viewing_item nodes, also_bought_item nodes, bought_together_item nodes. Brand nodes are linked to their item nodes.Please answer by providing node main feature (e.g., names) rather than node IDs.
        Question: What is the brand of item Professional Noise Cancelling Headset for Office Use? Please answer by providing node main feature (e.g., names) rather than node IDs.
        Thought 1: To answer the question, I need to find the node representing the item Professional Noise Cancelling Headset for Office Use and then check its associated brand node. First, I will retrieve the item node using the provided title.
        Action 1: Retrieve[Professional Noise Cancelling Headset for Office Use]
        Observation 1: The ID of this retrieval target node is B08XYZ1234.
        Thought 2: Now that I have the ID of the item node, I will check its brand neighbor to identify the brand associated with this item.
        Action 2: Neighbor[B08XYZ1234, brand]
        Observation 2: The brand neighbors of B08XYZ1234 are: ['brand_67890'].
        Thought 3: The brand associated with the item is represented by the node brand_67890. Next, I need to retrieve the name of this brand node.
        Action 3: Feature[brand_67890, name]
        Observation 3: The name feature of brand_67890 is ClearSound Tech.
        Thought 4: The brand name of the item is ClearSound Tech.
        Action 4: Finish[ClearSound Tech]
        Observation 4: Answer is ClearSound Tech

        Proceed with explanation and judgment below:
        Explanation: The final answer meets the problem requirements because the reasoning process is logical, supported by the observations, and directly corresponds to the data retrieved from the graph.
        Judgment: [Yes]

            """,
    "biomedical": """Solve a question answering task with interleaving Thought, Interaction with Graph, Feedback from Graph steps. In Thought step, you can think about what further information is needed, and In Interaction step, you can get feedback from graphs with four functions: 
        (1) Retrieve[keyword], which retrieves the related node from the graph according to the corresponding query.
        (2) Feature[Node, feature], which returns the detailed attribute information of Node regarding the given "feature" key.
        (3) Degree[Node, neighbor_type], which calculates the number of "neighbor_type" neighbors of the node Node in the graph.
        (4) Neighbor[Node, neighbor_type], which lists the "neighbor_type" neighbours of the node Node in the graph and returns them.
        Definition of the graph: There are eleven types of nodes in the graph: Anatomy, Biological Process, Cellular Component, Compound, Disease, Gene, Molecular Function, Pathway, Pharmacologic Class, Side Effect, Symptom.
        Each node has name feature.
        There are these types of edges: Anatomy-downregulates-Gene, Anatomy-expresses-Gene, Anatomy-upregulates-Gene, Compound-binds-Gene, Compound-causes-Side Effect, Compound-downregulates-Gene, Compound-palliates-Disease, Compound-resembles-Compound, Compound-treats-Disease, Compound-upregulates-Gene, Disease-associates-Gene, Disease-downregulates-Gene, Disease-localizes-Anatomy, Disease-presents-Symptom, Disease-resembles-Disease, Disease-upregulates-Gene, Gene-covaries-Gene, Gene-interacts-Gene, Gene-participates-Biological Process, Gene-participates-Cellular Component, Gene-participates-Molecular Function, Gene-participates-Pathway, Gene-regulates-Gene, Pharmacologic Class-includes-Compound.
        Definition of the graph: There are eleven types of nodes in the graph: Anatomy, Biological Process, Cellular Component, Compound, Disease, Gene, Molecular Function, Pathway, Pharmacologic Class, Side Effect, Symptom.
        Each node has name feature.
        There are these types of edges: Anatomy-downregulates-Gene, Anatomy-expresses-Gene, Anatomy-upregulates-Gene, Compound-binds-Gene, Compound-causes-Side Effect, Compound-downregulates-Gene, Compound-palliates-Disease, Compound-resembles-Compound, Compound-treats-Disease, Compound-upregulates-Gene, Disease-associates-Gene, Disease-downregulates-Gene, Disease-localizes-Anatomy, Disease-presents-Symptom, Disease-resembles-Disease, Disease-upregulates-Gene, Gene-covaries-Gene, Gene-interacts-Gene, Gene-participates-Biological Process, Gene-participates-Cellular Component, Gene-participates-Molecular Function, Gene-participates-Pathway, Gene-regulates-Gene, Pharmacologic Class-includes-Compound.
        Question: Could you tell me the side effects of the compound Aspirin? Please answer by providing node main feature (e.g., names) rather than node IDs.
        Thought 1: The question is related to a compound node (Aspirin). We need to find the node in the graph.
        Action 1: Retrieve[Aspirin]
        Observation 1: The ID of this retrieval target node is DB00945.
        Thought 2: The question is asking the side effects of the compound. We need to check the node's 'Compound-causes-Side Effect' neighbor from the graph.
        Action 2: Neighbor[DB00945, Compound-causes-Side Effect]
        Observation 2: The Compound-causes-Side Effect neighbors of DB00945 are: ['C1234567'].
        Thought 3: The side effect of Aspirin is C1234567.
        Action 3: Finish['C1234567']
        Observation 3: Answer is 'C1234567'

        Proceed with explanation and judgment below:
        Explanation: The final answer does not meet the problem requirements because the reasoning stops at retrieving the node ID of the side effect (C1234567) without retrieving its main feature (e.g., name). 
        Judgment: [No]

        Solve a question answering task with interleaving Thought, Interaction with Graph, Feedback from Graph steps. In Thought step, you can think about what further information is needed, and In Interaction step, you can get feedback from graphs with four functions: 
        (1) Retrieve[keyword], which retrieves the related node from the graph according to the corresponding query.
        (2) Feature[Node, feature], which returns the detailed attribute information of Node regarding the given "feature" key.
        (3) Degree[Node, neighbor_type], which calculates the number of "neighbor_type" neighbors of the node Node in the graph.
        (4) Neighbor[Node, neighbor_type], which lists the "neighbor_type" neighbours of the node Node in the graph and returns them.
        Definition of the graph: There are eleven types of nodes in the graph: Anatomy, Biological Process, Cellular Component, Compound, Disease, Gene, Molecular Function, Pathway, Pharmacologic Class, Side Effect, Symptom.
        Each node has name feature.
        There are these types of edges: Anatomy-downregulates-Gene, Anatomy-expresses-Gene, Anatomy-upregulates-Gene, Compound-binds-Gene, Compound-causes-Side Effect, Compound-downregulates-Gene, Compound-palliates-Disease, Compound-resembles-Compound, Compound-treats-Disease, Compound-upregulates-Gene, Disease-associates-Gene, Disease-downregulates-Gene, Disease-localizes-Anatomy, Disease-presents-Symptom, Disease-resembles-Disease, Disease-upregulates-Gene, Gene-covaries-Gene, Gene-interacts-Gene, Gene-participates-Biological Process, Gene-participates-Cellular Component, Gene-participates-Molecular Function, Gene-participates-Pathway, Gene-regulates-Gene, Pharmacologic Class-includes-Compound.
        Definition of the graph: There are eleven types of nodes in the graph: Anatomy, Biological Process, Cellular Component, Compound, Disease, Gene, Molecular Function, Pathway, Pharmacologic Class, Side Effect, Symptom.
        Each node has name feature.
        There are these types of edges: Anatomy-downregulates-Gene, Anatomy-expresses-Gene, Anatomy-upregulates-Gene, Compound-binds-Gene, Compound-causes-Side Effect, Compound-downregulates-Gene, Compound-palliates-Disease, Compound-resembles-Compound, Compound-treats-Disease, Compound-upregulates-Gene, Disease-associates-Gene, Disease-downregulates-Gene, Disease-localizes-Anatomy, Disease-presents-Symptom, Disease-resembles-Disease, Disease-upregulates-Gene, Gene-covaries-Gene, Gene-interacts-Gene, Gene-participates-Biological Process, Gene-participates-Cellular Component, Gene-participates-Molecular Function, Gene-participates-Pathway, Gene-regulates-Gene, Pharmacologic Class-includes-Compound.
        Question: What are the symptoms of the disease Systemic lupus erythematosus? Please answer by providing node main feature (e.g., names) rather than node IDs.
        Thought 1: The question is related to a disease node (Systemic lupus erythematosus). We need to find the node in the graph.
        Action 1: Retrieve[Systemic lupus erythematosus]
        Observation 1: The ID of this retrieval target node is DOID:9074.
        Thought 2: The question is asking the symptoms of the disease, we need to check the node's 'Disease-presents-Symptom' neighbor from the graph.
        Action 2: Neighbor[DOID:9074, Disease-presents-Symptom]
        Observation 2: The Disease-presents-Symptom neighbors of DOID:9074 are: ['D012889', 'D010146', 'D001943', 'D005865'].
        Thought 3: The symptoms of the disease are 'D012889', 'D010146', 'D001943', and 'D005865'.
        Action 3: Feature[D012889, name], Feature[D010146, name], Feature[D001943, name], Feature[D005865, name]
        Observation 3: The name feature of D012889 is: Malar Rash. The name feature of D010146 is: Photosensitivity. The name feature of D001943 is: Arthritis. The name feature of D005865 is: Hemolytic Anemia.
        Thought 4: The symptoms of the disease are Malar Rash, Photosensitivity, Arthritis, and Hemolytic Anemia.
        Action 4: Finish[Malar Rash, Photosensitivity, Arthritis, Hemolytic Anemia
        Observation 4: Answer is Malar Rash, Photosensitivity, Arthritis, Hemolytic Anemia

        Proceed with explanation and judgment below:
        Explanation: The final answer meets the problem requirements because the reasoning process is logical, supported by the observations, and directly corresponds to the data retrieved from the graph.
        Judgment: [Yes]
            """,
    "legal": """Solve a question answering task with interleaving Thought, Interaction with Graph, Feedback from Graph steps. In Thought step, you can think about what further information is needed, and In Interaction step, you can get feedback from graphs with four functions: 
        (1) Retrieve[keyword], which retrieves the related node from the graph according to the corresponding query.
        (2) Feature[Node, feature], which returns the detailed attribute information of Node regarding the given "feature" key.
        (3) Degree[Node, neighbor_type], which calculates the number of "neighbor_type" neighbors of the node Node in the graph.
        (4) Neighbor[Node, neighbor_type], which lists the "neighbor_type" neighbours of the node Node in the graph and returns them.
        Definition of the graph: There are four types of nodes in the graph: opinion, opinion_cluster, docket, and court.
        Opinion nodes have features: plain_text. Opinion_cluster nodes have features: syllabus, judges, case_name, attorneys. Docket nodes have features: pacer_case_id, case_name. Court nodes have features: full_name, start_date, end_date, citation_string.
        Opinion nodes are linked to their reference nodes and cited_by nodes, as well as their opinion_cluster nodes. Opinion_cluster nodes are linked to opinion nodes and docket nodes. Docket nodes are linked to opinion_cluster nodes and court nodes. Court nodes are linked to docket nodes.
        Question: which court is handling the case listed under the PACER docket number 108735.0? Please answer by providing node main feature (e.g., names) rather than node IDs.
        Thought 1: The question is asking for the court handling the case represented by the docket node with the given PACER case ID. We need to find the docket node first.
        Action 1: Retrieve[108735.0, Docket]
        Observation 1: The ID of this retrieval target node is d-65685551.
        Thought 2: Now that we have the docket node, we need to find the associated court node by checking its 'court' neighbor.
        Action 2: Neighbor[d-65685551, court]
        Observation 2: The court neighbors of d-65685551 are: ['ca9']. 
        Thought 3: The court that is handling the case is 'ca9'.
        Action 3: Finish[ca9]
        Observation 3: Answer is ca9

        Proceed with explanation and judgment below:
        Explanation: The final answer does not meet the problem requirements because it provides the node ID (ca9) of the court rather than the main feature (e.g., the court's name). 
        Judgment: [No]

        Solve a question answering task with interleaving Thought, Interaction with Graph, Feedback from Graph steps. In Thought step, you can think about what further information is needed, and In Interaction step, you can get feedback from graphs with four functions: 
        (1) Retrieve[keyword], which retrieves the related node from the graph according to the corresponding query.
        (2) Feature[Node, feature], which returns the detailed attribute information of Node regarding the given "feature" key.
        (3) Degree[Node, neighbor_type], which calculates the number of "neighbor_type" neighbors of the node Node in the graph.
        (4) Neighbor[Node, neighbor_type], which lists the "neighbor_type" neighbours of the node Node in the graph and returns them.
        Definition of the graph: There are four types of nodes in the graph: opinion, opinion_cluster, docket, and court.
        Opinion nodes have features: plain_text. Opinion_cluster nodes have features: syllabus, judges, case_name, attorneys. Docket nodes have features: pacer_case_id, case_name. Court nodes have features: full_name, start_date, end_date, citation_string.
        Opinion nodes are linked to their reference nodes and cited_by nodes, as well as their opinion_cluster nodes. Opinion_cluster nodes are linked to opinion nodes and docket nodes. Docket nodes are linked to opinion_cluster nodes and court nodes. Court nodes are linked to docket nodes.
        Question: What date does court Circuit Court, D. New York end on? Please answer by providing node main feature (e.g., names) rather than node IDs.
        Thought 1: The question is related to a court (Circuit Court, D. New York). We need to find the node in the graph.
        Action 1: Retrieve[Circuit Court, D. New York]
        Observation 1: The ID of this retrieval target node is newyorkd.
        Thought 2: The question is asking the end date of this court node, we need to check the node's 'end_date' feature from the graph.
        Action 2: Feature[newyorkd, end_date]
        Observation 2: The end_date feature of newyorkd is: 1891-03-03.
        Thought 3: The end date of this court is 1891-03-03.
        Action 3: Finish[1891-03-03]
        Observation 3: Answer is 1891-03-03
        
        Proceed with explanation and judgment below:
        Explanation: The final answer meets the problem requirements because the reasoning process is logical, supported by the observations, and directly corresponds to the data retrieved from the graph.
        Judgment: [Yes]
        """,
    "goodreads": """Solve a question answering task with interleaving Thought, Interaction with Graph, Feedback from Graph steps. In Thought step, you can think about what further information is needed, and In Interaction step, you can get feedback from graphs with four functions: 
        (1) Retrieve[keyword], which retrieves the related node from the graph according to the corresponding query.
        (2) Feature[Node, feature], which returns the detailed attribute information of Node regarding the given "feature" key.
        (3) Degree[Node, neighbor_type], which calculates the number of "neighbor_type" neighbors of the node Node in the graph.
        (4) Neighbor[Node, neighbor_type], which lists the "neighbor_type" neighbours of the node Node in the graph and returns them.
        Definition of the graph: There are four types of nodes in the graph: book, author, publisher, and series. 
        Book nodes have features: country_code, language_code, is_ebook, title, description, format, num_pages, publication_year, url, popular_shelves, and genres. Author nodes have features: name. Publisher nodes have features: name. Series nodes have features: title and description. 
        Book nodes are linked to their author nodes, publisher nodes, series nodes and similar_books nodes. Author nodes are linked to their book nodes. Publisher nodes are linked to their book nodes. Series nodes are linked to their book nodes.
        Question: What is the most common publication format of books by author Jane Austen? Please answer by providing node main feature (e.g., names) rather than node IDs.
        Thought 1: We need to find the author node for Jane Austen to determine the most common publication format of his books.
        Action 1: Retrieve[Jane Austen]
        Observation 1: The ID of this retrieval target node is author-105.
        Thought 2: Now that we have the author node, we need to calculate the node's book neighbor degree from the graph.
        Action 2: Degree[author-105, book]
        Observation 2: The book neighbor node degree of author-281126 are: 1. 
        Thought 3: The number of books is 1.
        Action 3: Finish[paperback]
        Observation 3: Answer is paperback

        Proceed with explanation and judgment below:
        Explanation: The final answer does not meet the problem requirements because it incorrectly assumes that the number of books directly determines the most common publication format without actually retrieving the format feature of the book nodes. 
        Judgment: [No]

        Solve a question answering task with interleaving Thought, Interaction with Graph, Feedback from Graph steps. In Thought step, you can think about what further information is needed, and In Interaction step, you can get feedback from graphs with four functions: 
        (1) Retrieve[keyword], which retrieves the related node from the graph according to the corresponding query.
        (2) Feature[Node, feature], which returns the detailed attribute information of Node regarding the given "feature" key.
        (3) Degree[Node, neighbor_type], which calculates the number of "neighbor_type" neighbors of the node Node in the graph.
        (4) Neighbor[Node, neighbor_type], which lists the "neighbor_type" neighbours of the node Node in the graph and returns them.
        Definition of the graph: There are four types of nodes in the graph: book, author, publisher, and series. 
        Book nodes have features: country_code, language_code, is_ebook, title, description, format, num_pages, publication_year, url, popular_shelves, and genres. Author nodes have features: name. Publisher nodes have features: name. Series nodes have features: title and description. 
        Book nodes are linked to their author nodes, publisher nodes, series nodes and similar_books nodes. Author nodes are linked to their book nodes. Publisher nodes are linked to their book nodes. Series nodes are linked to their book nodes.
        Question: Who are the authors of the book Echoes of the Arctic: Tales of Ice and Snow? Please answer by providing node main feature (e.g., names) rather than node IDs.
        Thought 1: The question is asking for the authors of a specific book (Echoes of the Arctic: Tales of Ice and Snow). We need to find the book node in the graph.
        Action 1: Retrieve[Echoes of the Arctic: Tales of Ice and Snow]
        Observation 1: The ID of this retrieval target node is book-8472619.
        Thought 2: The question is asking for the authors of the book. We need to check the book node's author neighbors from the graph.
        Action 2: Neighbor[book-8472619, author]
        Observation 2: The author neighbors of book-8472619 are: ['author-129874'].
        Thought 3: The IDs of the authors are 'author-129874'. We need to check the names.
        Action 3: Feature[author-129874, name]
        Observation 3: The name feature of author-129874 is: James Snowfield.
        Thought 4: The question is asking for all authors of the book, but we only have one author. We need to check if there are more authors.
        Action 4: Degree[book-8472619, author]
        Observation 4: The author neighbor node degree of book-8472619 is: 1.
        Thought 5: The number of author neighbors is 1. We can finish now.
        Action 5: Finish[James Snowfield]
        Observation 5: Answer is James Snowfield

        Proceed with explanation and judgment below:
        Explanation: The final answer meets the problem requirements because the reasoning process is logical, supported by the observations, and directly corresponds to the data retrieved from the graph.
        Judgment: [Yes]

        """,
    "dblp": """Solve a question answering task with interleaving Thought, Interaction with Graph, Feedback from Graph steps. In Thought step, you can think about what further information is needed, and In Interaction step, you can get feedback from graphs with four functions: 
        (1) Retrieve[keyword], which retrieves the related node from the graph according to the corresponding query.
        (2) Feature[Node, feature], which returns the detailed attribute information of Node regarding the given "feature" key.
        (3) Degree[Node, neighbor_type], which calculates the number of "neighbor_type" neighbors of the node Node in the graph.
        (4) Neighbor[Node, neighbor_type], which lists the "neighbor_type" neighbours of the node Node in the graph and returns them.
        Definition of the graph: There are three types of nodes in the graph: paper, author and venue.
        Paper nodes have features: title, abstract, keywords, lang, and year. Author nodes have features: name and organization. Venue nodes have features: name.
        Paper nodes are linked to their author nodes, venue nodes, reference nodes (the papers this paper cite) and cited_by nodes (other papers which cite this paper). Author nodes are linked to their paper nodes. Venue nodes are linked to their paper nodes.
        Definition of the graph: There are three types of nodes in the graph: paper, author and venue.
        Question: What is the number of papers that John Doe from Stanford University, Stanford, CA and Jane Smith from Stanford University, Palo Alto, CA co-authored? Please answer by providing node main feature (e.g., names) rather than node IDs.
        Thought 1: We need to find the nodes of these two researchers in the graph to determine their co-authored papers
        Action 1: Retrieve[John Doe], Retrieve[Jane Smith]
        Observation 1: The ID of John Doe's node is author-123456, and the ID of Jane Smith's node is author-789012.
        Thought 2: To confirm they are from the same organization, we need to check the organization feature of their nodes.
        Action 2: Feature[author-123456, organization], Feature[author-789012, organization]
        Observation 2: The organization of author-123456 is Stanford University, Stanford, CA. The organization of author-789012 is Stanford University, Palo Alto, CA.
        Thought 3: It's the same organization, now we can calculate the number of papers they co-authored.
        Action 3: Degree[author-123456, paper] / Degree[author-789012, paper]
        Observation 3: The paper neighbor node degree of author-123456 are: 1. The paper neighbor node degree of author-789012 are: 6. 
        Thought 4: There are six papers that they all write.
        Action 4: Finish[6]
        Observation 4: Answer is 6

        Proceed with explanation and judgment below:
        Explanation: The final answer does not meet the problem requirements because it incorrectly assumes that the Degree of each author represents the number of co-authored papers without verifying overlap between their paper neighbors. 
        Judgment: [No]

        Solve a question answering task with interleaving Thought, Interaction with Graph, Feedback from Graph steps. In Thought step, you can think about what further information is needed, and In Interaction step, you can get feedback from graphs with four functions: 
        (1) Retrieve[keyword], which retrieves the related node from the graph according to the corresponding query.
        (2) Feature[Node, feature], which returns the detailed attribute information of Node regarding the given "feature" key.
        (3) Degree[Node, neighbor_type], which calculates the number of "neighbor_type" neighbors of the node Node in the graph.
        (4) Neighbor[Node, neighbor_type], which lists the "neighbor_type" neighbours of the node Node in the graph and returns them.
        Definition of the graph: There are three types of nodes in the graph: paper, author and venue.
        Paper nodes have features: title, abstract, keywords, lang, and year. Author nodes have features: name and organization. Venue nodes have features: name.
        Paper nodes are linked to their author nodes, venue nodes, reference nodes (the papers this paper cite) and cited_by nodes (other papers which cite this paper). Author nodes are linked to their paper nodes. Venue nodes are linked to their paper nodes.
        Definition of the graph: There are three types of nodes in the graph: paper, author and venue.
        Question: Which institution is the researcher Elena Vargas Garcia associated with? Please answer by providing node main feature (e.g., names) rather than node IDs.
        Thought 1: The question is asking for the organization of a researcher named Elena Vargas Garcia. We need to find the node corresponding to Elena Vargas Garcia in the graph.
        Action 1: Retrieve[Elena Vargas Garcia]
        Observation 1: The ID of this retrieval target node is 62d8c97ab5439f87dc12345f.
        Thought 2: The question is asking for the organization of a researcher, so we need to check the node's organization feature from the graph.
        Action 2: Feature[62d8c97ab5439f87dc12345f, organization]
        Observation 2: The organization feature of 62d8c97ab5439f87dc12345f is: Department of Environmental Sciences, University of Valencia, C/ Dr. Moliner 50, 46100 Burjassot, Spain.
        Thought 3: The organization of researcher Elena Vargas Garcia is Department of Environmental Sciences, University of Valencia, C/ Dr. Moliner 50, 46100 Burjassot, Spain.
        Action 3: Finish[Department of Environmental Sciences, University of Valencia, C/ Dr. Moliner 50, 46100 Burjassot, Spain]
        Observation 3: Answer is Department of Environmental Sciences, University of Valencia, C/ Dr. Moliner 50, 46100 Burjassot, Spain

        Proceed with explanation and judgment below:
        Explanation: The final answer meets the problem requirements because the reasoning process is logical, supported by the observations, and directly corresponds to the data retrieved from the graph.
        Judgment: [Yes]

        """,
}

PLAN_SHORT_REFLECT_EXAMPLES={
"maple": """Previous trial Details:
Question: Who can be defined as the closest collaborator of author Alice B. Cooper, according to the number of their joint works? Please answer by providing node main feature (e.g., names) rather than node IDs.
plan 1: The question requires identifying the closest collaborator of "Alice B. Cooper" based on the number of joint works. Currently, we need to locate the node for Alice B. Cooper in the graph. 
Thought 1: We need to find the node for Alice B. Cooper in the graph.
Action 1: Retrieve[Alice B. Cooper]
Observation 1: The ID of this retrieval target node is 2676686768.
plan 2: The question requires determine the closest collaborator. We have the node ID of Alice B. Cooper. Thus, we need to count the number of collaborators.
Thought 2: We need to calculate the neighbor degree of this node to identify collaborators.
Action 2: Degree[2676686768, author]
Observation 2: The node or neighbor type does not exist in the graph. This might because your given neighbor type is not correct. Please modify it.
plan 3: The question is asking the closest collaborator of an author. We have the node ID of Alice B. Cooper. Since calculating the degree failed, we should explicitly check the author neighbors of this node and get their name.
Thought 3: We need to check the node's author neighbor from the graph.
Action 3: Feature[Neighbor[2676686768, author], name]
Observation 3: The node or neighbor type does not exist in the graph. This might because your given neighbor type is not correct. Please modify it.
plan 4: The question is asking the closest collaborator of an author, we are unable to retrieve the closest collaborator.
Thought 4: It seems that there is an issue with the graph structure or the given neighbor type, due to issues with the neighbor type.
Action 4: Finish[Unable to retrieve closest collaborator]
Observation 4: Answer is Unable to retrieve closest collaborator

Reflection:
- Recap of the Trial
  - Question: Who can be defined as the closest collaborator of author Alice B. Cooper, according to the number of their joint works? Please answer by providing node main feature (e.g., names) rather than node IDs.
  - Graph Information Used:
    Retrieve[Alice B. Cooper]: To find the node representing Alice B. Cooper.
    Degree[2676686768, author]: To calculate the degree of "author" neighbors.
    Feature[Neighbor[2676686768, author], name]: To retrieve the names of all "author" neighbors.
  - Outcome: The question was not successfully answered due to incorrect assumptions about the graph structure and the failure of neighbor-related functions.
- Guided Reflection
  - Understanding the Question
    - Core Goal: Identify the closest collaborator of Alice B. Cooper by evaluating the number of joint works, which requires correctly analyzing the relationships between nodes in the graph.
    - Missed Information: Misinterpreted or overlooked the correct relationship or neighbor type that represents "collaborators."
    - Potential Misunderstandings: Misinterpreted the graph's structure, assuming direct "author" neighbors represent collaborators. The collaboration is mediated by shared "paper" nodes.
  - Analysis of Selected Graph Information
    - Relevance: Retrieve[Alice B. Cooper]: Correct step to locate the node representing the author. Degree[2676686768, author]:Misapplied, as direct "author" neighbors do not exist. Feature[Neighbor[2676686768, author], name]: Irrelevant, as there are no direct "author" neighbors to retrieve.
    - Missed Insights: Did not consider exploring the graph to confirm how collaborators are connected (e.g., through shared papers). Overlooked the use of Feature to extract details about "paper" connections.
    - Redundancies: Repeated attempts to use "author" as the neighbor type without validating its relevance were redundant.
  - Align the Problem Understanding and Graph Function Information
    - Understanding and Selection: The misunderstanding of the question's requirements led to inappropriate graph function calls.
    - Inconsistencies: Assumed collaborators were directly connected as "author" neighbors, which was inconsistent with the graph's definition.
    - Adjustments: Start by understanding the graph's structure and how nodes are linked (e.g., "author" nodes are linked through "paper" nodes). Verify the relationship structure of the graph first using exploratory functions or metadata queries.
- Improved Strategy
  - Updated Understanding of the Problem:
    Collaborators of Alice B. Cooper are determined by shared "paper" nodes. To find the closest collaborator, it is necessary to identify the "paper" neighbors of Alice B. Cooper, then determine the authors with the highest co-authorship count.
  - Revised Graph Selection:
    - Graph Selection
    Use Retrieve[Alice B. Cooper] to find Alice B. Cooper's node.
    Use Neighbor with the type "paper" to list all "paper" neighbors of Alice B. Cooper.
    For each "paper" node, check its "author" neighbors using Neighbor.
    Count the occurrences of each author node in the "paper" neighbors list to find the collaborator with the highest joint works.
    Use Feature to retrieve the name of the closest collaborator.
    - Combine Functions suggestions
    You can combine Retrieve with Neighbor to list all papers authored by Alice B. Cooper: Neighbor[Retrieve[Alice B. Cooper], paper]
  - Avoiding Past Issues: 
    Confirm graph structure and neighbor relationships before performing specific queries.
    Avoid redundant actions by verifying assumptions at each step.
    Use appropriate graph functions (e.g., Neighbor and Feature) to extract detailed relationship data.(END OF REFLECTION)\n\n\n\n
    (END OF REFLECTION)
            """,
"amazon": """Previous trial Details:
Question: What is the count of co-viewed items for item Columbia Women's Silver Ridge Lite Long Sleeve Shirt? Please answer by providing node main feature (e.g., names) rather than node IDs.
Plan 1: The question is asking the count of co-viewed items for a item node (Columbia Women's Silver Ridge Lite Long Sleeve Shirt), we need to find the node first.
Thought 1: We need to find the node in the graph.
Action 1: Retrieve[Columbia Women's Silver Ridge Lite Long Sleeve Shirt]
Observation 1: The ID of this retrieval target node is ABCDE00000.
Plan 2: The question is asking the count of co-viewed items for an item. We have the node ID of Columbia Women's Silver Ridge Lite Long Sleeve Shirt. Thus, we need to calculate the count of node's also_viewed_item neighbors.
Thought 2: We need to calculate the node's also_viewed_item neighbor degree from the graph.
Action 2: Degree[ABCDE00000, also_viewed_item]
Observation 2: The also_viewed_item neighbor node degree of ABCDE00000 are: 15. 
Plan3:  The question is asking the count of co-viewed items for an item. We have the node ID of the item and its co-view count, so we need to list the node's also_viewed_item neighbors to identify the co-viewed items by their features.
Thought 3: We need to list the node's also_viewed_item neighbors from the graph.
Action 3: Neighbor[ABCDE00000, also_viewed_item]
Observation 3: The also_viewed_item neighbors of ABCDE00000 are: ['B009XNJPX8', 'B00FAN8CF2', 'B0043EVJJA', 'B009XNJOBQ', 'B0043EVIVE'].
Plan 4: The question is asking for the count of co-viewed items. We have the co-viewed items listed, so we need to provide them.
Thought 4:The co-viewed items for item Columbia Women's Silver Ridge Lite Long Sleeve Shirt are 'B009XNJPX8', 'B00FAN8CF2', 'B0043EVJJA', 'B009XNJOBQ', 'B0043EVIVE'.
Action 4:Finish['B009XNJPX8', 'B00FAN8CF2', 'B0043EVJJA', 'B009XNJOBQ', 'B0043EVIVE']
Observation 4: Answer is 'B009XNJPX8', 'B00FAN8CF2', 'B0043EVJJA', 'B009XNJOBQ', 'B0043EVIVE'

Reflection:
- Recap of the Trial
  - Question: What is the count of co-viewed items for the item Columbia Women's Silver Ridge Lite Long Sleeve Shirt? Please answer by providing node main features (e.g., names) rather than node IDs.
  - Graph Information Used:
    Retrieve[Columbia Women's Silver Ridge Lite Long Sleeve Shirt]: To locate the node representing Columbia Women's Silver Ridge Lite Long Sleeve Shirt.
    Degree[ABCDE00000, also_viewed_item]: To calculate the degree of also_viewed_item neighbors for Columbia Women's Silver Ridge Lite Long Sleeve Shirt.
    Neighbor[ABCDE00000, also_viewed_item]: To list neighbors of type also_viewed_item for Columbia Women's Silver Ridge Lite Long Sleeve Shirt.
  - Outcome: The question was not successfully answered because of guessing the wrong answer with Finish[].
- Guided Reflection
  -  Understanding the Question
    - Core Goal: Provide the count of also_viewed_item neighbors for the specified item.
    - Missed Information: Misunderstood the requirement to provide only the count of neighbors, leading to an unnecessary listing of neighbor details.
    - Potential Misunderstandings: Assumed the question required listing the names of also_viewed_item neighbors instead of focusing on the numeric count.
  - Analysis of Selected Graph Information
    - Relevance: Retrieve: Correct step to locate the item node. Degree: Appropriate to calculate the degree of also_viewed_item neighbors, directly addressing the question. Neighbor: Redundant in this context since the question did not require listing the neighbors.
    - Missed Insights: Overlooked that Degree alone could answer the question by providing the required count.
    - Redundancies: The additional step of Neighbor was unnecessary and complicated the process.
  - Align the Problem Understanding and Graph Function Information
    - Understanding and Selection: Misaligned the use of Neighbor, assuming additional details were required.
    - Inconsistencies: Misunderstood the question's requirement, leading to redundant steps.
    - Adjustments: Focus on using Degree to calculate and return the count without additional queries.
- Improved Strategy
  - Updated Understanding of the Problem:
    The question requires the numeric count of also_viewed_item neighbors for the specified item. Listing neighbor details is unnecessary.
  - Revised Graph Selection:
    - Graph Selection
    Use Retrieve[Columbia Women's Silver Ridge Lite Long Sleeve Shirt] to locate the target node.
    Use Degree with the type also_viewed_item to calculate and directly return the count of co-viewed items.
    - Combine Functions suggestions
    You can combine Retrieve with Degree to directly return the count of co-viewed items for Columbia Women's Silver Ridge Lite Long Sleeve Shirt: Degree[Retrieve[Columbia Women's Silver Ridge Lite Long Sleeve Shirt], also_viewed_item]
  - Avoiding Past Issues: 
    Focus on the core requirement (providing the count).
    Eliminate redundant queries like Neighbor unless explicitly needed.
    Use concise and relevant graph operations to align with the problem's objective.(END OF REFLECTION)\n\n\n\n
    (END OF REFLECTION)
            """,
"biomedical": """Previous trial Details:
Question: Could you tell me the side effects of the compound Aspirin? Please answer by providing node main feature (e.g., names) rather than node IDs.
Plan 1: The question is asking for the side effects of the compound Aspirin. We need to locate the node for Aspirin.
Thought 1: We need to find the node in the graph.
Action 1: Retrieve[Aspirin]
Observation 1: The ID of this retrieval target node is DB00945.
Plan 2: The question is asking for the side effects of the compound Aspirin. We have the node ID of Aspirin as DB00945. Thus, we need to check the node's Compound-causes-Side Effect neighbors in the graph.
Thought 2: We need to check the node's 'Compound-causes-Side Effect' neighbor from the graph.
Action 2: Neighbor[DB00945, Compound-causes-Side Effect]
Observation 2: The Compound-causes-Side Effect neighbors of DB00945 are: ['C1234567'].
Plan 3: The question is asking for the side effects of the compound Aspirin. We have the Compound-causes-Side Effect neighbor. Thus, we need to identify the side effects and return them as the answer.
Thought 3: The side effect of Aspirin is C1234567.
Action 3: Finish['C1234567']
Observation 3: Answer is 'C1234567'

Reflection: 
- Recap of the Trial
  - Question: What are the side effects of the compound Aspirin? Please answer by providing node main feature (e.g., names) rather than node IDs.
  - Graph Information Used:
    Retrieve[Aspirin]: To locate the node representing Aspirin.
    Neighbor[DB00945, Compound-causes-Side Effect]: To retrieve neighbors of type Compound-causes-Side Effect.
  - Outcome: The question was not successfully answered because of guessing the wrong answer with Finish[].
- Guided Reflection
  -  Understanding the Question
    - Core Goal: Identify and provide the names of all side effects caused by Aspirin using the Compound-causes-Side Effect relationship.
    - Missed Information: Failed to retrieve the name feature of the side effect node.
    - Potential Misunderstandings: Assumed that the side effect node ID was sufficient as an answer, overlooking the need to query and provide the node's name feature.
  - Analysis of Selected Graph Information
    - Relevance: Retrieve: Correct to locate the compound node for Aspirin. Neighbor: Correct to identify the side effect nodes linked to Aspirin.
    - Missed Insights: Did not use Feature to retrieve the name of the side effect node, which is essential to answer the question accurately.
    - Redundancies: None; all actions were directly related to answering the question, but an additional step to fetch the name was required.
  - Align the Problem Understanding and Graph Function Information
    - Understanding and Selection: Misinterpreted the requirement to provide the node's name feature, leading to an incomplete answer.
    - Inconsistencies: Assumed that providing the node ID sufficed, which was inconsistent with the question's explicit requirement.
    - Adjustments: Add a step to query the name feature of each retrieved side effect node.
- Improved Strategy
  - Updated Understanding of the Problem:
    The question requires retrieving the names of all side effects caused by Aspirin.
  - Revised Graph Selection:
    - Graph Selection
    Use Retrieve[Aspirin] to locate the compound node.
    Use Neighbor[DB00945, Compound-causes-Side Effect] to retrieve all side effect nodes.
    Use Feature on each retrieved side effect node to extract their name features.
    - Combine Functions suggestions
    Use Retrieve[Aspirin] to locate the compound node. 
    Combine Neighbor with Feature to directly return the name features of side effect nodes: Feature[Neighbor[DB00945, Compound-causes-Side Effect], name]
  - Avoiding Past Issues: 
    Ensure that all answers are provided in terms of the node name feature as required.(END OF REFLECTION)\n\n\n\n
    (END OF REFLECTION)
            """,
"legal": """Previous trial Details:
Question: which court is handling the case listed under the PACER docket number 108735.0? Please answer by providing node main feature (e.g., names) rather than node IDs.
Plan 1: The question is asking for the court handling the case listed under the PACER docket number 108735.0. We need to find the corresponding docket node in the graph. 
Thought 1: We need to find the docket node first.
Action 1: Retrieve[108735.0, Docket]
Observation 1: The ID of this retrieval target node is d-65685551.
Plan 2: The question is asking for the court handling the case represented by the docket node. We have the docket node ID as d-65685551. Thus, we need to check its court neighbor in the graph to identify the associated court.
Thought 2: We need to find the associated court node by checking its 'court' neighbor.
Action 2: Neighbor[d-65685551, court]
Observation 2: The court neighbors of d-65685551 are: ['ca9']. 
Plan 3: The question is asking for the court handling the case. We have identified the court neighbor. Thus, we need to provide the name of the court as the answer.
Thought 3: The court that is handling the case is 'ca9'.
Action 3: Finish[ca9]
Observation 3: Answer is ca9

Reflection: 
- Recap of the Trial
  - Question: Which court is handling the case listed under the PACER docket number 108735.0? Please answer by providing node main feature (e.g., names) rather than node IDs.
  - Graph Information Used:
    Retrieve[108735.0, Docket]: To locate the docket node corresponding to the PACER case ID.
    Neighbor[d-65685551, court]: To retrieve the court node linked to the docket node.
  - Outcome: The question was not successfully answered because of guessing the wrong answer with Finish[].
- Guided Reflection
  -  Understanding the Question
    - Core Goal: Identify the name or main feature of the court handling the case, rather than its node ID.
    - Missed Information: Did not extract the court node's full_name feature, which is the required detail for answering the question correctly.
    - Potential Misunderstandings: Assumed the court node ID (ca9) was sufficient, overlooking the need to query and return the full_name feature.
  - Analysis of Selected Graph Information
    - Relevance: Retrieve: Correctly identified the docket node. Neighbor: Correctly found the court node linked to the docket.
    - Missed Insights: Failed to query the Feature for the full_name of the court node, which was explicitly required to answer the question accurately.
    - Redundancies: None; the steps taken were necessary but incomplete.
  - Align the Problem Understanding and Graph Function Information
    - Understanding and Selection: Correctly identified the relationship between docket and court nodes but failed to retrieve the necessary feature of the court node.
    - Inconsistencies: Provided the node ID (ca9) instead of the required full_name attribute.
    - Adjustments: Add a step to query the Feature of the court node for the full_name attribute.
- Improved Strategy
  - Updated Understanding of the Problem:
    The question requires identifying the court handling the case by providing its full_name attribute, not its node ID.
  - Revised Graph Selection:
    - Graph Selection
    Use Retrieve[108735.0, Docket] to locate the docket node for the PACER case ID.
    Use Neighbor[d-65685551, court] to find the court node linked to the docket.
    Use Feature[ca9, full_name] to retrieve the court's full_name feature.
    - Combine Functions suggestions
    Use Retrieve[108735.0, Docket] to locate the docket node for the PACER case ID.
    Combine Neighbor with Feature to directly return the name features of the court linked to the docket : Feature[Neighbor[d-65685551, court], full_name]
  - Avoiding Past Issues: 
    Ensure the response includes the requested feature (full_name) rather than raw node IDs.(END OF REFLECTION)\n\n\n\n
    (END OF REFLECTION)

        """,
"goodreads": """Previous trial Details:
Question: What is the most common publication format of books by author Jane Austen? Please answer by providing node main feature (e.g., names) rather than node IDs.
Plan 1: The question is asking for the most common publication format of books by the author Jane Austen. We have the author's name. Thus, we need to find the corresponding author node in the graph.
Thought 1: We need to find the author node for Jane Austen.
Action 1: Retrieve[Jane Austen]
Observation 1: The ID of this retrieval target node is author-105.
Plan 2: The question is asking for the publication format of books written by Jane Austen. We have the author node ID as author-105. Thus, we need to calculate the degree of its book neighbors to determine how many books are connected to this author.
Thought 2: We need to calculate the node's book neighbor degree from the graph.
Action 2: Degree[author-105, book]
Observation 2: The book neighbor node degree of author-281126 are: 1. 
Plan 3: The question is asking for the most common publication format of books by Jane Austen. We have determined there is only 1 book, so we need to identify its publication format and provide it as the answer.
Thought 3: The number of books is 1.
Action 3: Finish[paperback]
Observation 3: Answer is paperback

Reflection: 
- Recap of the Trial
  - Question: What is the most common publication format of books by author Jane Austen? Please answer by providing node main feature (e.g., names) rather than node IDs.
  - Graph Information Used:
    Retrieve[Jane Austen]: Found the author node.
    Degree[author-105, book]: Checked the degree of book neighbors for the author node.
  - Outcome: The question was not successfully answered because of guessing the wrong answer with Finish[].
- Guided Reflection
  -  Understanding the Question
    - Core Goal: Determine the most common publication format of books written by Jane Austen by analyzing the format feature of the linked book nodes.
    - Missed Information: Failed to retrieve the format feature of the related book nodes.
    - Potential Misunderstandings: Guessed the format (paperback) without extracting data from the graph.
  - Analysis of Selected Graph Information
    - Relevance: Retrieve: Correctly identified the author node. Degree: Checked the number of book nodes but was not sufficient to answer the question.
    - Missed Insights: Did not retrieve the list of book nodes (Neighbor) for further analysis. Failed to query the format feature of the book nodes to determine the most common format.
    - Redundancies: The use of Degree without follow-up queries to analyze the data was unnecessary.
  - Align the Problem Understanding and Graph Function Information
    - Understanding and Selection: Misunderstood the requirement to retrieve and analyze the format feature of books, leading to incomplete reasoning.
    - Inconsistencies: Provided an answer (paperback) without validating it through the graph data.
    - Adjustments: Ensure the retrieval of relevant features (e.g., format) from all related nodes.
- Improved Strategy
  - Updated Understanding of the Problem:
    The question requires identifying the most common format of books written by Jane Austen by analyzing the format feature of all related book nodes.
  - Revised Graph Selection:
    - Graph Selection
    Use Retrieve[Jane Austen] to locate the author node.
    Use Neighbor[author-105, book] to retrieve all book nodes linked to the author.
    Use Feature on each retrieved book node to extract the format feature.
    Analyze the retrieved format values to determine the most common format.
    - Combine Functions suggestions
    Use Retrieve[Jane Austen] to locate the author node.
    Combine Neighbor with Feature to directly return the format feature of each retrieved book written by Jane Austen: Feature[Neighbor[author-105, book], format]
    Analyze the retrieved format values to determine the most common format.
  - Avoiding Past Issues: 
    Do not guess the answer without verifying data through appropriate queries.
    Retrieve and analyze all relevant features of the book nodes.(END OF REFLECTION)\n\n\n\n
    (END OF REFLECTION)
        """,
"dblp": """Previous trial Details:
Question: What is the number of papers that John Doe from Stanford University, Stanford, CA and Jane Smith from Stanford University, Palo Alto, CA co-authored? Please answer by providing node main feature (e.g., names) rather than node IDs.
Plan 1: The question is asking for the number of papers co-authored by John Doe from Stanford University, Stanford, CA, and Jane Smith from Stanford University, Palo Alto, CA. We need to find the corresponding nodes in the graph for each. 
Thought 1: We need to find the nodes of these two researchers in the graph.
Action 1: Retrieve[John Doe], Retrieve[Jane Smith]
Observation 1: The ID of John Doe's node is author-123456, and the ID of Jane Smith's node is author-789012.
Plan 2: The question is asking to confirm whether both researchers are from the same organization. We have their node IDs. Thus, we need to check the organization feature of their respective nodes.
Thought 2: We need to check the organization feature of their nodes.
Action 2: Feature[author-123456, organization], Feature[author-789012, organization]
Observation 2: The organization of author-123456 is Stanford University, Stanford, CA. The organization of author-789012 is Stanford University, Palo Alto, CA.
Plan 3: The question is asking for the number of papers co-authored by these researchers. We have confirmed they are from the same organization. Thus, we need to calculate the number of papers both have authored and identify the overlap (co-authored papers).
Thought 3: We can calculate the number of papers they co-authored.
Action 3: Degree[author-123456, paper] / Degree[author-789012, paper]
Observation 3: The paper neighbor node degree of author-123456 are: 1. The paper neighbor node degree of author-789012 are: 6. 
Plan 4: The question is asking for the number of papers co-authored by the two researchers. We have determined their individual paper counts. Thus, we need to identify the overlap (co-authored papers) and provide the final count.
Thought 4: There are six papers that they all write.
Action 4: Finish[6]
Observation 4: Answer is 6

Reflection: 
- Recap of the Trial
  - Question: What is the number of papers that John Doe from Stanford University, Stanford, CA and Jane Smith from Stanford University, Palo Alto, CA co-authored? Please answer by providing node main feature (e.g., names) rather than node IDs.
  - Graph Information Used:
    Retrieve[John Doe] and Retrieve[Jane Smith]: Located the nodes for the two authors.
    Feature[author-123456, organization] and Feature[author-789012, organization]: Confirmed they are from the same organization.
    Degree[author-123456, paper] and Degree[author-789012, paper]: Attempted to calculate the number of co-authored papers.
  - Outcome: The question was not successfully answered because of guessing the wrong answer with Finish[].
- Guided Reflection
  -  Understanding the Question
    - Core Goal: Determine the exact number of papers co-authored by John Doe and Jane Smith by finding their common paper neighbors in the graph.
    - Missed Information: Failed to check the Neighbor for both authors to identify common papers.
    - Potential Misunderstandings: Incorrectly interpreted the individual Degree values as the number of co-authored papers.
  - Analysis of Selected Graph Information
    - Relevance: Retrieve: Correctly identified the nodes for the two authors. Feature: Appropriately used to confirm the organization of the authors. Degree: Misused as a proxy for co-authored paper count.
    - Missed Insights: Did not retrieve and compare the actual paper neighbors to identify overlap. Overlooked the need to verify co-authorship explicitly through Neighbor.
    - Redundancies: Checking the node degree for individual authors without retrieving the actual neighbor lists was redundant.
  - Align the Problem Understanding and Graph Function Information
    - Understanding and Selection: The misunderstanding of how node degrees relate to co-authorship caused the failure.
    - Inconsistencies:  Failed to align the goal of finding shared paper nodes with the graph functions available for exploring relationships.
    - Adjustments: Use Neighbor to retrieve the paper neighbors for each author. Compare the retrieved neighbor lists to find overlap, accurately reflecting the number of co-authored papers.
- Improved Strategy
  - Updated Understanding of the Problem:
    The question requires identifying the number of papers co-authored by two authors by comparing their shared paper neighbors.
  - Revised Graph Selection:
    - Graph Selection
    Use Retrieve[John Doe] and Retrieve[Jane Smith] to locate the nodes for the authors.
    Use Feature to confirm their organizational affiliations if needed for context.
    Use Neighbor[author-123456, paper] and Neighbor[author-789012, paper] to retrieve their respective paper neighbors.
    Compare the lists of paper neighbors to identify common nodes (co-authored papers).
    Count the number of common papers and provide the result.
    - Combine Functions suggestions
    None. Each graph function step must be conducted separately to ensure accurate identification of shared paper.
  - Avoiding Past Issues: 
    Avoid relying on Degree as a measure of shared connections without verification.
    Ensure neighbor relationships are explicitly retrieved and analyzed for overlap.(END OF REFLECTION)\n\n\n\n
    (END OF REFLECTION)
        """,
        }

PLAN_SHORT_EVAL_EXAMPLES={
    "maple": """Solve a question answering task with interleaving Thought, Interaction with Graph, Feedback from Graph steps. In Plan step, you can think about what the question is asking and plan how to do to get the answer. In Thought step, you can think about what further information is needed, and In Interaction step, you can get feedback from graphs with four functions: 
        (1) Retrieve[keyword], which retrieves the related node from the graph according to the corresponding query.
        (2) Feature[Node, feature], which returns the detailed attribute information of Node regarding the given "feature" key.
        (3) Degree[Node, neighbor_type], which calculates the number of "neighbor_type" neighbors of the node Node in the graph.
        (4) Neighbor[Node, neighbor_type], which lists the "neighbor_type" neighbours of the node Node in the graph and returns them.
        Besides, you can use compound function, such as Feature[Retrieve[keyword], feature], which returns the detailed attribute information of Retrieve[keyword] regarding the given "feature" key.
        When last Observation has been given or there is no Plan, you should provide next only one Plan based on the question. When last Plan has been given, you should provide next only one Thought. When last Thought has been given, you should provide next only one Action. When you think it's time to finish, use Finish to end the process. Don't make Observation.
        Question: Whom has the author Emily T. Anderson often been working with? Please answer by providing node main feature (e.g., names) rather than node IDs.
        plan 1:  The question is asking for co-authors of Emily T. Anderson. Thus, we need to locate the relevant author node in the graph.
        Thought 1: We need to find the node representing Emily T. Anderson in the graph.
        Action 1: Retrieve[Emily T. Anderson]
        Observation 1: The ID of this retrieval target node is 3456789123.
        plan 2: The question is asking for co-authors of Emily T. Anderson. To find co-authors, we need to identify the papers authored by Emily T. Anderson.
        Thought 2:  We need to retrieve the paper neighbors of the author node.
        Action 2: Neighbor[3456789123, paper]
        Observation 2: The paper neighbors of 3456789123 are: ['4012345678', '4098765432']. 
        plan 3: The question is asking for co-authors of Emily T. Anderson. Having identified the papers authored by Emily T. Anderson, we need to retrieve the authors of these papers to find her co-authors.
        Thought 3: We should check the author neighbors of each paper node.
        Action 3: Neighbor[4012345678, author], Neighbor[4098765432, author]
        Observation 3: The author neighbors of paper 4012345678 are: ['3456789123', '2233445566', '3344556677'].The author neighbors of paper 4098765432 are: ['3456789123', '5566778899', '2233445577'].
        plan 4: The question is asking for co-authors of Emily T. Anderson. After collecting the co-author nodes, we need to retrieve their names from the graph.
        Thought 4: We now have a list of co-authors. Next, we need to retrieve their names.
        Action 4: Feature[3456789123, name], Feature[2233445566, name], Feature[3344556677, name], Feature[3456789123, name], Feature[5566778899, name], Feature[2233445577, name]
        Observation 4: The name feature of 3456789123 are: John D. Miller. The name feature of 2233445566 are: Sarah K. Williams. The name feature of 3344556677 are: Mark A. Johnson. The name feature of 3456789123 are: John D. Miller. The name feature of 5566778899 are: Anna M. Green. The name feature of 2233445577 are: David P. Brown. 
        plan 5: The question is asking for co-authors of Emily T. Anderson, we need compile the names of the co-authors into the final answer.
        Thought 5: The co-authors of Emily T. Anderson are John D. Miller, Sarah K. Williams, Mark A. Johnson,  Anna M. Green, and David P. Brown.
        Action 5: Finish[John D. Miller, Sarah K. Williams, Mark A. Johnson,  Anna M. Green, and David P. Brown]
        Observation 5: Answer is John D. Miller, Sarah K. Williams, Mark A. Johnson,  Anna M. Green, and David P. Brown

        Proceed with explanation and judgment below:
        Explanation: The process identified the papers authored by Emily T. Anderson and retrieved the associated co-authors correctly. However, there was a logical misstep in Thought 4, where the question required identifying the co-authors Emily T. Anderson often works with. This implies we need to determine the most frequent collaborators.
        Judgment: [No]

        Solve a question answering task with interleaving Thought, Interaction with Graph, Feedback from Graph steps. In Plan step, you can think about what the question is asking and plan how to do to get the answer. In Thought step, you can think about what further information is needed, and In Interaction step, you can get feedback from graphs with four functions: 
        (1) Retrieve[keyword], which retrieves the related node from the graph according to the corresponding query.
        (2) Feature[Node, feature], which returns the detailed attribute information of Node regarding the given "feature" key.
        (3) Degree[Node, neighbor_type], which calculates the number of "neighbor_type" neighbors of the node Node in the graph.
        (4) Neighbor[Node, neighbor_type], which lists the "neighbor_type" neighbours of the node Node in the graph and returns them.
        Besides, you can use compound function, such as Feature[Retrieve[keyword], feature], which returns the detailed attribute information of Retrieve[keyword] regarding the given "feature" key.
        When last Observation has been given or there is no Plan, you should provide next only one Plan based on the question. When last Plan has been given, you should provide next only one Thought. When last Thought has been given, you should provide next only one Action. When you think it's time to finish, use Finish to end the process. Don't make Observation.
        Question: Whom has the author Emily T. Anderson often been working with? Please answer by providing node main feature (e.g., names) rather than node IDs.
        plan 1: The question is asking for the co-author who has collaborated with Emily T. Anderson the most. We need to locate the relevant author node in the graph, get the papers written by her and find the most frequent co-author.
        Thought 1: We need to find the node representing Emily T. Anderson in the graph.
        Action 1: Retrieve[Emily T. Anderson]
        Observation 1: The ID of this retrieval target node is 3456789123.
        plan 2: The question is asking for the co-author who has collaborated with Emily T. Anderson the most. We have the author's node. Thus, we need to identify the papers authored by Emily T. Anderson and find the most frequent co-author.
        Thought 2:  We need to retrieve the paper neighbors of the author node and get the authors of these papers.
        Action 2: Neighbor[Neighbor[3456789123, paper], author]
        Observation 2: The paper neighbors of 3456789123 are: ['4012345678', '4098765432']. The author neighbors of paper 4012345678 are: ['3456789123', '2233445566', '3344556677'].The author neighbors of paper 4098765432 are: ['3456789123', '5566778899', '2233445577'].
        plan 3: The question is asking for the co-author who has collaborated with Emily T. Anderson the most. We have the lists of co-authors. Thus, we need to find the most frequent node in the two lists to find the most frequent co-author.
        Thought 3: We find the most frequent co-author is 3456789123. We need to check the name feature of the co-author.
        Action 3: Feature[3456789123, name]
        Observation 3: The name feature of 3456789123 is: John D. Miller.
        plan 4: The question is asking for the co-author who has collaborated with Emily T. Anderson the most. We get the answer -- John D. Miller.
        Thought 4: The most frequent co-authors of Emily T. Anderson is John D. Miller.
        Action 4: Finish[John D. Miller]
        Observation 5: Answer is John D. Miller

        Proceed with explanation and judgment below:
        Explanation: The final answer meets the problem requirements because the reasoning process is logical, supported by the observations, and directly corresponds to the data retrieved from the graph.
        Judgment: [Yes]
        """,
        "amazon":"""Solve a question answering task with interleaving Thought, Interaction with Graph, Feedback from Graph steps. In Thought step, you can think about what further information is needed, and In Interaction step, you can get feedback from graphs with four functions: 
        (1) Retrieve[keyword], which retrieves the related node from the graph according to the corresponding query.
        (2) Feature[Node, feature], which returns the detailed attribute information of Node regarding the given "feature" key.
        (3) Degree[Node, neighbor_type], which calculates the number of "neighbor_type" neighbors of the node Node in the graph.
        (4) Neighbor[Node, neighbor_type], which lists the "neighbor_type" neighbours of the node Node in the graph and returns them.
        Besides, you can use compound function, such as Feature[Retrieve[keyword], feature], which returns the detailed attribute information of Retrieve[keyword] regarding the given "feature" key.
        When last Observation has been given or there is no Plan, you should provide next only one Plan based on the question. When last Plan has been given, you should provide next only one Thought. When last Thought has been given, you should provide next only one Action. When you think it's time to finish, use Finish to end the process. Don't make Observation.
        Definition of the graph: There are two types of nodes in the graph: item and brand.
        Item nodes have features: title, description, price, img, category. Brand nodes have features: name.
        Item nodes are linked to their brand nodes, also_viewed_item nodes, buy_after_viewing_item nodes, also_bought_item nodes, bought_together_item nodes. Brand nodes are linked to their item nodes.Please answer by providing node main feature (e.g., names) rather than node IDs.
        Question: Which item shares over 4 co-viewed items with the item Healthy Home Cooking: Nutritious Recipes for Everyday Meals? Please answer by providing node main feature (e.g., names) rather than node IDs.
        Plan 1: The question is asking for an item that shares more than 4 co-viewed items with the item Healthy Home Cooking: Nutritious Recipes for Everyday Meals. We need to find the ID of the item and then check the number of its also_viewed_item neighbors.
        Thought 1: We need to find the ID of the item.
        Action 1: Retrieve[Healthy Home Cooking: Nutritious Recipes for Everyday Meals]
        Observation 1: The ID of this retrieval target node is 123456789.
        Plan 2: The question is asking for an item that shares more than 4 co-viewed items with the item "Healthy Home Cooking: Nutritious Recipes for Everyday Meals". We have the ID of the item. Thus, we need to check the number of its also_viewed_item neighbors.
        Thought 2: We need to find its also_viewed_item neighbors.
        Action 2: Neighbor[123456789, also_viewed_item]
        Observation 2: The also_viewed_item neighbors of 123456789 are: ['158040250X', '0807847909', '0688162193', '0806520280', '0688100120', '0738212288', '146960762X', '0809231530', '0767903145', '0671796291', '0802132839', '1580087981', '0060187166', '1607745313', '0399146563', '161243097X', '1580402275', '080785655X', '159240085X', '1439193304', '1580400752', '0764569112', '1469608707', '0806526777']. 
        Plan 3: The question is asking for an item that shares more than 4 co-viewed items with the item "Healthy Home Cooking: Nutritious Recipes for Everyday Meals". We have the also_viewed_item neighbors of the item. Thus, we need to determine if any of these also_viewed_item neighbors share more than 4 co-viewed items with the original item.
        Thought 3: We need to determine if any of these also_viewed_item neighbors share more than 4 co-viewed items with the original item.
        Action 3: Degree[158040250X, also_viewed_item]
        Observation 3: The also_viewed_item neighbor node degree of 158040250X are: 5. 
        Plan 4: The question is asking for an item that shares more than 4 co-viewed items with the item "Healthy Home Cooking: Nutritious Recipes for Everyday Meals". We have the answer -- 158040250X.
        Thought 4: The node degree of 158040250X is 5, which means it shares more than 4 co-viewed items with the original item.
        Action 4: Finish[158040250X]
        Observation 4: Answer is 158040250X

        Proceed with explanation and judgment below:
        Explanation: The answer assumes that the node degree represents shared co-viewed items without verifying overlap through a comparison of neighbor lists.
        Judgment: [No]

        Solve a question answering task with interleaving Thought, Interaction with Graph, Feedback from Graph steps. In Thought step, you can think about what further information is needed, and In Interaction step, you can get feedback from graphs with four functions: 
        (1) Retrieve[keyword], which retrieves the related node from the graph according to the corresponding query.
        (2) Feature[Node, feature], which returns the detailed attribute information of Node regarding the given "feature" key.
        (3) Degree[Node, neighbor_type], which calculates the number of "neighbor_type" neighbors of the node Node in the graph.
        (4) Neighbor[Node, neighbor_type], which lists the "neighbor_type" neighbours of the node Node in the graph and returns them.
        Besides, you can use compound function, such as Feature[Retrieve[keyword], feature], which returns the detailed attribute information of Retrieve[keyword] regarding the given "feature" key.
        When last Observation has been given or there is no Plan, you should provide next only one Plan based on the question. When last Plan has been given, you should provide next only one Thought. When last Thought has been given, you should provide next only one Action. When you think it's time to finish, use Finish to end the process. Don't make Observation.
        Definition of the graph: There are two types of nodes in the graph: item and brand.
        Item nodes have features: title, description, price, img, category. Brand nodes have features: name.
        Item nodes are linked to their brand nodes, also_viewed_item nodes, buy_after_viewing_item nodes, also_bought_item nodes, bought_together_item nodes. Brand nodes are linked to their item nodes.Please answer by providing node main feature (e.g., names) rather than node IDs.
        Question: What is the brand of item Professional Noise Cancelling Headset for Office Use? Please answer by providing node main feature (e.g., names) rather than node IDs.
        Plan 1: The question is asking for the brand of the item Professional Noise Cancelling Headset for Office Use. We need to find the ID of the item and then check its brand neighbor.
        Thought 1: We need to retrieve the item node using the provided title.
        Action 1: Retrieve[Professional Noise Cancelling Headset for Office Use]
        Observation 1: The ID of this retrieval target node is B08XYZ1234.
        Plan 2: The question is asking for the brand of the item Professional Noise Cancelling Headset for Office Use. We have the ID of the item. Thus, we need to check its brand neighbor and get its name.
        Thought 2: We need to check its brand neighbor to identify the brand associated with this item and get its name.
        Action 2: Feature[Neighbor[B08XYZ1234, brand], name]
        Observation 2: The brand neighbors of B08XYZ1234 are: ['brand_67890']. The name feature of brand_67890 is ClearSound Tech.
        Plan 3: The question is asking for the brand of the item Professional Noise Cancelling Headset for Office Use. We get the answer -- ClearSound Tech.
        Thought 3: The brand name of the item is ClearSound Tech.
        Action 3: Finish[ClearSound Tech]
        Observation 3: Answer is ClearSound Tech

        Proceed with explanation and judgment below:
        Explanation: The final answer meets the problem requirements because the reasoning process is logical, supported by the observations, and directly corresponds to the data retrieved from the graph.
        Judgment: [Yes]
            """,

    "biomedical": """Solve a question answering task with interleaving Thought, Interaction with Graph, Feedback from Graph steps. In Thought step, you can think about what further information is needed, and In Interaction step, you can get feedback from graphs with four functions: 
        (1) Retrieve[keyword], which retrieves the related node from the graph according to the corresponding query.
        (2) Feature[Node, feature], which returns the detailed attribute information of Node regarding the given "feature" key.
        (3) Degree[Node, neighbor_type], which calculates the number of "neighbor_type" neighbors of the node Node in the graph.
        (4) Neighbor[Node, neighbor_type], which lists the "neighbor_type" neighbours of the node Node in the graph and returns them.
        Besides, you can use compound function, such as Feature[Retrieve[keyword], feature], which returns the detailed attribute information of Retrieve[keyword] regarding the given "feature" key.
        When last Observation has been given or there is no Plan, you should provide next only one Plan based on the question. When last Plan has been given, you should provide next only one Thought. When last Thought has been given, you should provide next only one Action. When you think it's time to finish, use Finish to end the process. Don't make Observation.
        Definition of the graph: There are eleven types of nodes in the graph: Anatomy, Biological Process, Cellular Component, Compound, Disease, Gene, Molecular Function, Pathway, Pharmacologic Class, Side Effect, Symptom.
        Each node has name feature.
        There are these types of edges: Anatomy-downregulates-Gene, Anatomy-expresses-Gene, Anatomy-upregulates-Gene, Compound-binds-Gene, Compound-causes-Side Effect, Compound-downregulates-Gene, Compound-palliates-Disease, Compound-resembles-Compound, Compound-treats-Disease, Compound-upregulates-Gene, Disease-associates-Gene, Disease-downregulates-Gene, Disease-localizes-Anatomy, Disease-presents-Symptom, Disease-resembles-Disease, Disease-upregulates-Gene, Gene-covaries-Gene, Gene-interacts-Gene, Gene-participates-Biological Process, Gene-participates-Cellular Component, Gene-participates-Molecular Function, Gene-participates-Pathway, Gene-regulates-Gene, Pharmacologic Class-includes-Compound.
        Question: Could you tell me the side effects of the compound Aspirin? Please answer by providing node main feature (e.g., names) rather than node IDs.
        Plan 1: The question is asking for the side effects of a compound node (Aspirin). We need to find the node in the graph and get its side effects.
        Thought 1: We need to find the node in the graph.
        Action 1: Retrieve[Aspirin]
        Observation 1: The ID of this retrieval target node is DB00945.
        Plan 2: The question is asking for the side effects of a compound node (Aspirin). We have the node ID. Thus, we need to check its side effects.
        Thought 2: We need to check the node's 'Compound-causes-Side Effect' neighbor from the graph.
        Action 2: Neighbor[DB00945, Compound-causes-Side Effect]
        Observation 2: The Compound-causes-Side Effect neighbors of DB00945 are: ['C1234567'].
        Plan 3: The question is asking for the side effects of a compound node (Aspirin). We get the answer -- C1234567.
        Thought 3: The side effect of Aspirin is C1234567.
        Action 3: Finish['C1234567']
        Observation 3: Answer is 'C1234567'

        Proceed with explanation and judgment below:
        Explanation: The final answer does not meet the problem requirements because the reasoning stops at retrieving the node ID of the side effect (C1234567) without retrieving its main feature (e.g., name). 
        Judgment: [No]

        Solve a question answering task with interleaving Thought, Interaction with Graph, Feedback from Graph steps. In Thought step, you can think about what further information is needed, and In Interaction step, you can get feedback from graphs with four functions: 
        (1) Retrieve[keyword], which retrieves the related node from the graph according to the corresponding query.
        (2) Feature[Node, feature], which returns the detailed attribute information of Node regarding the given "feature" key.
        (3) Degree[Node, neighbor_type], which calculates the number of "neighbor_type" neighbors of the node Node in the graph.
        (4) Neighbor[Node, neighbor_type], which lists the "neighbor_type" neighbours of the node Node in the graph and returns them.
        Besides, you can use compound function, such as Feature[Retrieve[keyword], feature], which returns the detailed attribute information of Retrieve[keyword] regarding the given "feature" key.
        When last Observation has been given or there is no Plan, you should provide next only one Plan based on the question. When last Plan has been given, you should provide next only one Thought. When last Thought has been given, you should provide next only one Action. When you think it's time to finish, use Finish to end the process. Don't make Observation.
        Definition of the graph: There are eleven types of nodes in the graph: Anatomy, Biological Process, Cellular Component, Compound, Disease, Gene, Molecular Function, Pathway, Pharmacologic Class, Side Effect, Symptom.
        Each node has name feature.
        There are these types of edges: Anatomy-downregulates-Gene, Anatomy-expresses-Gene, Anatomy-upregulates-Gene, Compound-binds-Gene, Compound-causes-Side Effect, Compound-downregulates-Gene, Compound-palliates-Disease, Compound-resembles-Compound, Compound-treats-Disease, Compound-upregulates-Gene, Disease-associates-Gene, Disease-downregulates-Gene, Disease-localizes-Anatomy, Disease-presents-Symptom, Disease-resembles-Disease, Disease-upregulates-Gene, Gene-covaries-Gene, Gene-interacts-Gene, Gene-participates-Biological Process, Gene-participates-Cellular Component, Gene-participates-Molecular Function, Gene-participates-Pathway, Gene-regulates-Gene, Pharmacologic Class-includes-Compound.
        Question: What are the symptoms of the disease Systemic lupus erythematosus? Please answer by providing node main feature (e.g., names) rather than node IDs.
        Plan 1: The question is asking for the symptoms of a disease node (Systemic lupus erythematosus). We need to find the node in the graph and get its symptoms.
        Thought 1: We need to find the node in the graph.
        Action 1: Retrieve[Systemic lupus erythematosus]
        Observation 1: The ID of this retrieval target node is DOID:9074.
        Plan 2: The question is asking for the symptoms of a disease node (Systemic lupus erythematosus). We have the node ID. Thus, we need to check its symptoms and get its name.
        Thought 2: We need to check the node's 'Disease-presents-Symptom' neighbor from the graph and get its name.
        Action 2: Feature[Neighbor[DOID:9074, Disease-presents-Symptom], name]
        Observation 2: The Disease-presents-Symptom neighbors of DOID:9074 are: ['D012889', 'D010146', 'D001943', 'D005865']. The name feature of D012889 is: Malar Rash. The name feature of D010146 is: Photosensitivity. The name feature of D001943 is: Arthritis. The name feature of D005865 is: Hemolytic Anemia.
        Plan 3: The question is asking for the symptoms of a disease node (Systemic lupus erythematosus). We get the answer -- Malar Rash, Photosensitivity, Arthritis, Hemolytic Anemia.
        Thought 3: The symptoms of the disease are Malar Rash, Photosensitivity, Arthritis, and Hemolytic Anemia.
        Action 3: Finish[Malar Rash, Photosensitivity, Arthritis, Hemolytic Anemia
        Observation 3: Answer is Malar Rash, Photosensitivity, Arthritis, Hemolytic Anemia

        Proceed with explanation and judgment below:
        Explanation: The final answer meets the problem requirements because the reasoning process is logical, supported by the observations, and directly corresponds to the data retrieved from the graph.
        Judgment: [Yes]
            """,
    "legal": """Solve a question answering task with interleaving Thought, Interaction with Graph, Feedback from Graph steps. In Thought step, you can think about what further information is needed, and In Interaction step, you can get feedback from graphs with four functions: 
        (1) Retrieve[keyword], which retrieves the related node from the graph according to the corresponding query.
        (2) Feature[Node, feature], which returns the detailed attribute information of Node regarding the given "feature" key.
        (3) Degree[Node, neighbor_type], which calculates the number of "neighbor_type" neighbors of the node Node in the graph.
        (4) Neighbor[Node, neighbor_type], which lists the "neighbor_type" neighbours of the node Node in the graph and returns them.
        Besides, you can use compound function, such as Feature[Retrieve[keyword], feature], which returns the detailed attribute information of Retrieve[keyword] regarding the given "feature" key.
        When last Observation has been given or there is no Plan, you should provide next only one Plan based on the question. When last Plan has been given, you should provide next only one Thought. When last Thought has been given, you should provide next only one Action. When you think it's time to finish, use Finish to end the process. Don't make Observation.
        Definition of the graph: There are four types of nodes in the graph: opinion, opinion_cluster, docket, and court.
        Opinion nodes have features: plain_text. Opinion_cluster nodes have features: syllabus, judges, case_name, attorneys. Docket nodes have features: pacer_case_id, case_name. Court nodes have features: full_name, start_date, end_date, citation_string.
        Opinion nodes are linked to their reference nodes and cited_by nodes, as well as their opinion_cluster nodes. Opinion_cluster nodes are linked to opinion nodes and docket nodes. Docket nodes are linked to opinion_cluster nodes and court nodes. Court nodes are linked to docket nodes.
        Question: which court is handling the case listed under the PACER docket number 108735.0? Please answer by providing node main feature (e.g., names) rather than node IDs.
        Plan 1: The question is asking for the court handling the case listed under the PACER docket number 108735.0. We need to find the docket node and get its court.
        Thought 1: We need to find the docket node first.
        Action 1: Retrieve[108735.0, Docket]
        Observation 1: The ID of this retrieval target node is d-65685551.
        Plan 2: The question is asking for the court handling the case listed under the PACER docket number 108735.0. We have the docket node. Thus, we need to check its court.
        Thought 2: We need to find the associated court node by checking its 'court' neighbor.
        Action 2: Neighbor[d-65685551, court]
        Observation 2: The court neighbors of d-65685551 are: ['ca9']. 
        Plan 3: The question is asking for the court handling the case listed under the PACER docket number 108735.0. We get the answer -- ca9.
        Thought 3: The court that is handling the case is 'ca9'.
        Action 3: Finish[ca9]
        Observation 3: Answer is ca9

        Proceed with explanation and judgment below:
        Explanation: The final answer does not meet the problem requirements because it provides the node ID (ca9) of the court rather than the main feature (e.g., the court's name). 
        Judgment: [No]

        Solve a question answering task with interleaving Thought, Interaction with Graph, Feedback from Graph steps. In Thought step, you can think about what further information is needed, and In Interaction step, you can get feedback from graphs with four functions: 
        (1) Retrieve[keyword], which retrieves the related node from the graph according to the corresponding query.
        (2) Feature[Node, feature], which returns the detailed attribute information of Node regarding the given "feature" key.
        (3) Degree[Node, neighbor_type], which calculates the number of "neighbor_type" neighbors of the node Node in the graph.
        (4) Neighbor[Node, neighbor_type], which lists the "neighbor_type" neighbours of the node Node in the graph and returns them.
        Besides, you can use compound function, such as Feature[Retrieve[keyword], feature], which returns the detailed attribute information of Retrieve[keyword] regarding the given "feature" key.
        When last Observation has been given or there is no Plan, you should provide next only one Plan based on the question. When last Plan has been given, you should provide next only one Thought. When last Thought has been given, you should provide next only one Action. When you think it's time to finish, use Finish to end the process. Don't make Observation.
        Definition of the graph: There are four types of nodes in the graph: opinion, opinion_cluster, docket, and court.
        Opinion nodes have features: plain_text. Opinion_cluster nodes have features: syllabus, judges, case_name, attorneys. Docket nodes have features: pacer_case_id, case_name. Court nodes have features: full_name, start_date, end_date, citation_string.
        Opinion nodes are linked to their reference nodes and cited_by nodes, as well as their opinion_cluster nodes. Opinion_cluster nodes are linked to opinion nodes and docket nodes. Docket nodes are linked to opinion_cluster nodes and court nodes. Court nodes are linked to docket nodes.
        Question: What date does court Circuit Court, D. New York end on? Please answer by providing node main feature (e.g., names) rather than node IDs.
        Plan 1: The question is asking for the end date of a court node (Circuit Court, D. New York). We need to find the node in the graph and get its end date.
        Thought 1: We need to find the node in the graph and get its end date.
        Action 1: Feature[Retrieve[Circuit Court, D. New York], end_date]
        Observation 1: The ID of this retrieval target node is newyorkd. The end_date feature of newyorkd is: 1891-03-03.
        Plan 2: The question is asking for the end date of a court node (Circuit Court, D. New York). We get the answer -- 1891-03-03.
        Thought 2: The end date of this court is 1891-03-03.
        Action 2: Finish[1891-03-03]
        Observation 2: Answer is 1891-03-03
        
        Proceed with explanation and judgment below:
        Explanation: The final answer meets the problem requirements because the reasoning process is logical, supported by the observations, and directly corresponds to the data retrieved from the graph.
        Judgment: [Yes]
            """,
    "goodreads": """Solve a question answering task with interleaving Thought, Interaction with Graph, Feedback from Graph steps. In Thought step, you can think about what further information is needed, and In Interaction step, you can get feedback from graphs with four functions: 
        (1) Retrieve[keyword], which retrieves the related node from the graph according to the corresponding query.
        (2) Feature[Node, feature], which returns the detailed attribute information of Node regarding the given "feature" key.
        (3) Degree[Node, neighbor_type], which calculates the number of "neighbor_type" neighbors of the node Node in the graph.
        (4) Neighbor[Node, neighbor_type], which lists the "neighbor_type" neighbours of the node Node in the graph and returns them.
        Besides, you can use compound function, such as Feature[Retrieve[keyword], feature], which returns the detailed attribute information of Retrieve[keyword] regarding the given "feature" key.
        When last Observation has been given or there is no Plan, you should provide next only one Plan based on the question. When last Plan has been given, you should provide next only one Thought. When last Thought has been given, you should provide next only one Action. When you think it's time to finish, use Finish to end the process. Don't make Observation.
        Definition of the graph: There are four types of nodes in the graph: book, author, publisher, and series. 
        Book nodes have features: country_code, language_code, is_ebook, title, description, format, num_pages, publication_year, url, popular_shelves, and genres. Author nodes have features: name. Publisher nodes have features: name. Series nodes have features: title and description. 
        Book nodes are linked to their author nodes, publisher nodes, series nodes and similar_books nodes. Author nodes are linked to their book nodes. Publisher nodes are linked to their book nodes. Series nodes are linked to their book nodes.
        Question: What is the most common publication format of books by author Jane Austen? Please answer by providing node main feature (e.g., names) rather than node IDs.
        Plan 1: The question is asking for the most common publication format of books by author Jane Austen. We need to find the author node and get its most common publication format.
        Thought 1: We need to find the author node for Jane Austen to determine the most common publication format of his books.
        Action 1: Retrieve[Jane Austen]
        Observation 1: The ID of this retrieval target node is author-105.
        Plan 2: The question is asking for the most common publication format of books by author Jane Austen. We have the ID of the author. Thus, we need to calculate the node's book neighbor degree from the graph.
        Thought 2: We need to calculate the node's book neighbor degree from the graph.
        Action 2: Degree[author-105, book]
        Observation 2: The book neighbor node degree of author-281126 are: 1. 
        Plan 3: The question is asking for the most common publication format of books by author Jane Austen. We get the answer -- 1.
        Thought 3: The number of books is 1.
        Action 3: Finish[paperback]
        Observation 3: Answer is paperback

        Proceed with explanation and judgment below:
        Explanation: The final answer does not meet the problem requirements because it incorrectly assumes that the number of books directly determines the most common publication format without actually retrieving the format feature of the book nodes. 
        Judgment: [No]

        Solve a question answering task with interleaving Thought, Interaction with Graph, Feedback from Graph steps. In Thought step, you can think about what further information is needed, and In Interaction step, you can get feedback from graphs with four functions: 
        (1) Retrieve[keyword], which retrieves the related node from the graph according to the corresponding query.
        (2) Feature[Node, feature], which returns the detailed attribute information of Node regarding the given "feature" key.
        (3) Degree[Node, neighbor_type], which calculates the number of "neighbor_type" neighbors of the node Node in the graph.
        (4) Neighbor[Node, neighbor_type], which lists the "neighbor_type" neighbours of the node Node in the graph and returns them.
        Besides, you can use compound function, such as Feature[Retrieve[keyword], feature], which returns the detailed attribute information of Retrieve[keyword] regarding the given "feature" key.
        When last Observation has been given or there is no Plan, you should provide next only one Plan based on the question. When last Plan has been given, you should provide next only one Thought. When last Thought has been given, you should provide next only one Action. When you think it's time to finish, use Finish to end the process. Don't make Observation.
        Definition of the graph: There are four types of nodes in the graph: book, author, publisher, and series. 
        Book nodes have features: country_code, language_code, is_ebook, title, description, format, num_pages, publication_year, url, popular_shelves, and genres. Author nodes have features: name. Publisher nodes have features: name. Series nodes have features: title and description. 
        Book nodes are linked to their author nodes, publisher nodes, series nodes and similar_books nodes. Author nodes are linked to their book nodes. Publisher nodes are linked to their book nodes. Series nodes are linked to their book nodes.
        Question: Who are the authors of the book Echoes of the Arctic: Tales of Ice and Snow? Please answer by providing node main feature (e.g., names) rather than node IDs.
        Plan 1: The question is asking for the authors of a specific book (Echoes of the Arctic: Tales of Ice and Snow). We need to find the book node in the graph and check its author neighbors.
        Thought 1: We need to find the book node in the graph.
        Action 1: Retrieve[Echoes of the Arctic: Tales of Ice and Snow]
        Observation 1: The ID of this retrieval target node is book-8472619.
        Plan 2: The question is asking for the authors of the book. We have the book node. Thus, we need to check the book node's author neighbors from the graph and retrieve the author's name.
        Thought 2: We need to check the book node's author neighbors from the graph and retrieve the author's name.
        Action 2: Feature[Neighbor[book-8472619, author], name]
        Observation 2: The author neighbors of book-8472619 are: ['author-129874']. The name feature of author-129874 is: James Snowfield.
        Plan 3: The question is asking for the authors of the book. We have the author's name. Thus, we need to make sure there is only one author.
        Thought 3: We need to check if there are more authors.
        Action 3: Degree[book-8472619, author]
        Observation 3: The author neighbor node degree of book-8472619 is: 1.
        Plan 4: The question is asking for the authors of the book. We get the answer -- James Snowfield.
        Thought 4: The number of author neighbors is 1. We can finish now.
        Action 4: Finish[James Snowfield]
        Observation 4: Answer is James Snowfield

        Proceed with explanation and judgment below:
        Explanation: The final answer meets the problem requirements because the reasoning process is logical, supported by the observations, and directly corresponds to the data retrieved from the graph.
        Judgment: [Yes]
        """,
    "dblp": """Solve a question answering task with interleaving Thought, Interaction with Graph, Feedback from Graph steps. In Thought step, you can think about what further information is needed, and In Interaction step, you can get feedback from graphs with four functions: 
        (1) Retrieve[keyword], which retrieves the related node from the graph according to the corresponding query.
        (2) Feature[Node, feature], which returns the detailed attribute information of Node regarding the given "feature" key.
        (3) Degree[Node, neighbor_type], which calculates the number of "neighbor_type" neighbors of the node Node in the graph.
        (4) Neighbor[Node, neighbor_type], which lists the "neighbor_type" neighbours of the node Node in the graph and returns them.
        Besides, you can use compound function, such as Feature[Retrieve[keyword], feature], which returns the detailed attribute information of Retrieve[keyword] regarding the given "feature" key.
        When last Observation has been given or there is no Plan, you should provide next only one Plan based on the question. When last Plan has been given, you should provide next only one Thought. When last Thought has been given, you should provide next only one Action. When you think it's time to finish, use Finish to end the process. Don't make Observation.
        Definition of the graph: There are three types of nodes in the graph: paper, author and venue.
        Paper nodes have features: title, abstract, keywords, lang, and year. Author nodes have features: name and organization. Venue nodes have features: name.
        Paper nodes are linked to their author nodes, venue nodes, reference nodes (the papers this paper cite) and cited_by nodes (other papers which cite this paper). Author nodes are linked to their paper nodes. Venue nodes are linked to their paper nodes.
        Definition of the graph: There are three types of nodes in the graph: paper, author and venue.
        Question: What is the number of papers that John Doe from Stanford University, Stanford, CA and Jane Smith from Stanford University, Palo Alto, CA co-authored? Please answer by providing node main feature (e.g., names) rather than node IDs.
        Plan 1: The question is asking for the number of papers that two researchers co-authored. We need to find the nodes of these two researchers in the graph and get their co-authored papers.
        Thought 1: We need to find the nodes of these two researchers in the graph to determine their co-authored papers
        Action 1: Retrieve[John Doe], Retrieve[Jane Smith]
        Observation 1: The ID of John Doe's node is author-123456, and the ID of Jane Smith's node is author-789012.
        Plan 2: The question is asking for the number of papers that two researchers co-authored. We have the nodes of these two researchers. Thus, we need to calculate the number of papers they co-authored.
        Thought 2: To confirm they are from the same organization, we need to check the organization feature of their nodes.
        Action 2: Feature[author-123456, organization], Feature[author-789012, organization]
        Observation 2: The organization of author-123456 is Stanford University, Stanford, CA. The organization of author-789012 is Stanford University, Palo Alto, CA.
        Plan 3: The question is asking for the number of papers that two researchers co-authored. We konw the organization is the same. Thus, we can check the number of papers they co-authored.
        Thought 3: We can calculate the number of papers they co-authored.
        Action 3: Degree[author-123456, paper] / Degree[author-789012, paper]
        Observation 3: The paper neighbor node degree of author-123456 are: 1. The paper neighbor node degree of author-789012 are: 6. 
        Plan 4: The question is asking for the number of papers that two researchers co-authored. We get the answer -- 6.
        Thought 4: There are six papers that they all write.
        Action 4: Finish[6]
        Observation 4: Answer is 6

        Proceed with explanation and judgment below:
        Explanation: The final answer does not meet the problem requirements because it incorrectly assumes that the Degree of each author represents the number of co-authored papers without verifying overlap between their paper neighbors. 
        Judgment: [No]

        Solve a question answering task with interleaving Thought, Interaction with Graph, Feedback from Graph steps. In Thought step, you can think about what further information is needed, and In Interaction step, you can get feedback from graphs with four functions: 
        (1) Retrieve[keyword], which retrieves the related node from the graph according to the corresponding query.
        (2) Feature[Node, feature], which returns the detailed attribute information of Node regarding the given "feature" key.
        (3) Degree[Node, neighbor_type], which calculates the number of "neighbor_type" neighbors of the node Node in the graph.
        (4) Neighbor[Node, neighbor_type], which lists the "neighbor_type" neighbours of the node Node in the graph and returns them.
        Besides, you can use compound function, such as Feature[Retrieve[keyword], feature], which returns the detailed attribute information of Retrieve[keyword] regarding the given "feature" key.
        When last Observation has been given or there is no Plan, you should provide next only one Plan based on the question. When last Plan has been given, you should provide next only one Thought. When last Thought has been given, you should provide next only one Action. When you think it's time to finish, use Finish to end the process. Don't make Observation.
        Definition of the graph: There are three types of nodes in the graph: paper, author and venue.
        Paper nodes have features: title, abstract, keywords, lang, and year. Author nodes have features: name and organization. Venue nodes have features: name.
        Paper nodes are linked to their author nodes, venue nodes, reference nodes (the papers this paper cite) and cited_by nodes (other papers which cite this paper). Author nodes are linked to their paper nodes. Venue nodes are linked to their paper nodes.
        Definition of the graph: There are three types of nodes in the graph: paper, author and venue.
        Question: Which institution is the researcher Elena Vargas Garcia associated with? Please answer by providing node main feature (e.g., names) rather than node IDs.
        Plan 1: The question is asking for the organization of a researcher named Elena Vargas Garcia. We need to find the node corresponding to Elena Vargas Garcia in the graph and check its organization.
        Thought 1: We need to find the node corresponding to Elena Vargas Garcia in the graph and check its organization.
        Action 1: Feature[Retrieve[Elena Vargas Garcia], organization]
        Observation 1: The ID of this retrieval target node is 62d8c97ab5439f87dc12345f. The organization feature of 62d8c97ab5439f87dc12345f is: Department of Environmental Sciences, University of Valencia, C/ Dr. Moliner 50, 46100 Burjassot, Spain.
        Plan 2: The question is asking for the organization of a researcher. We get the answer -- Department of Environmental Sciences, University of Valencia, C/ Dr. Moliner 50, 46100 Burjassot, Spain.
        Thought 2: The organization of researcher Elena Vargas Garcia is Department of Environmental Sciences, University of Valencia, C/ Dr. Moliner 50, 46100 Burjassot, Spain.
        Action 2: Finish[Department of Environmental Sciences, University of Valencia, C/ Dr. Moliner 50, 46100 Burjassot, Spain]
        Observation 2: Answer is Department of Environmental Sciences, University of Valencia, C/ Dr. Moliner 50, 46100 Burjassot, Spain

        Proceed with explanation and judgment below:
        Explanation: The final answer meets the problem requirements because the reasoning process is logical, supported by the observations, and directly corresponds to the data retrieved from the graph.
        Judgment: [Yes]
        """,
    "biomedical_compound": """Definition of the graph: There are eleven types of nodes in the graph: Anatomy, Biological Process, Cellular Component, Compound, Disease, Gene, Molecular Function, Pathway, Pharmacologic Class, Side Effect, Symptom.
        Each node has name feature.
        There are these types of edges: Anatomy-downregulates-Gene, Anatomy-expresses-Gene, Anatomy-upregulates-Gene, Compound-binds-Gene, Compound-causes-Side Effect, Compound-downregulates-Gene, Compound-palliates-Disease, Compound-resembles-Compound, Compound-treats-Disease, Compound-upregulates-Gene, Disease-associates-Gene, Disease-downregulates-Gene, Disease-localizes-Anatomy, Disease-presents-Symptom, Disease-resembles-Disease, Disease-upregulates-Gene, Gene-covaries-Gene, Gene-interacts-Gene, Gene-participates-Biological Process, Gene-participates-Cellular Component, Gene-participates-Molecular Function, Gene-participates-Pathway, Gene-regulates-Gene, Pharmacologic Class-includes-Compound.
        Question: What compounds can be used to treat Crohn's disease? Please answer the compound names rather than IDs.
        Thought 1: The question is related to a disease node (Crohn's disease). We need to find the node in the graph.
        Action 1: Retrieve[Crohn's disease]
        Observation 1: The ID of this node is DOID:8778.
        Thought 2: The question is asking the compounds which can be used to treat a disease, we need to check the node's 'Compound-treats-Disease' neighbor from the graph and get their names.
        Action 2: Feature[Neighbor[DOID:8778, Compound-treats-Disease], name]
        Observation 2: ['DB01014', 'DB00244', 'DB00795', 'DB00993', 'DB00635', 'DB01033'] Balsalazide, Balsalazide, Mesalazine, Sulfasalazine, Azathioprine, Prednisone, Mercaptopurine
        Thought 3: The name of compounds are Balsalazide, Mesalazine, Sulfasalazine, Azathioprine, Prednisone, Mercaptopurine.
        Action 3: Finish[Balsalazide, Mesalazine, Sulfasalazine, Azathioprine, Prednisone, Mercaptopurine]
        
        Definition of the graph: There are eleven types of nodes in the graph: Anatomy, Biological Process, Cellular Component, Compound, Disease, Gene, Molecular Function, Pathway, Pharmacologic Class, Side Effect, Symptom.
        Each node has name feature.
        There are these types of edges: Anatomy-downregulates-Gene, Anatomy-expresses-Gene, Anatomy-upregulates-Gene, Compound-binds-Gene, Compound-causes-Side Effect, Compound-downregulates-Gene, Compound-palliates-Disease, Compound-resembles-Compound, Compound-treats-Disease, Compound-upregulates-Gene, Disease-associates-Gene, Disease-downregulates-Gene, Disease-localizes-Anatomy, Disease-presents-Symptom, Disease-resembles-Disease, Disease-upregulates-Gene, Gene-covaries-Gene, Gene-interacts-Gene, Gene-participates-Biological Process, Gene-participates-Cellular Component, Gene-participates-Molecular Function, Gene-participates-Pathway, Gene-regulates-Gene, Pharmacologic Class-includes-Compound.
        Question: What is the inchikey of Caffeine?
        Thought 1: The question is related to a compound node (Caffeine). We need to find the node in the graph and check the node's 'inchikey' feature from the graph.
        Action 1: Feature[Retrieve[Caffeine], inchikey]
        Observation 1: The ID of this node is DB00201. InChIKey=RYYVLZVUVIJVGH-UHFFFAOYSA-N
        Thought 2:  The inchikey of the node is InChIKey=RYYVLZVUVIJVGH-UHFFFAOYSA-N.
        Action 2: Finish[InChIKey=RYYVLZVUVIJVGH-UHFFFAOYSA-N]
        
        Definition of the graph: There are eleven types of nodes in the graph: Anatomy, Biological Process, Cellular Component, Compound, Disease, Gene, Molecular Function, Pathway, Pharmacologic Class, Side Effect, Symptom.
        Each node has name feature.
        There are these types of edges: Anatomy-downregulates-Gene, Anatomy-expresses-Gene, Anatomy-upregulates-Gene, Compound-binds-Gene, Compound-causes-Side Effect, Compound-downregulates-Gene, Compound-palliates-Disease, Compound-resembles-Compound, Compound-treats-Disease, Compound-upregulates-Gene, Disease-associates-Gene, Disease-downregulates-Gene, Disease-localizes-Anatomy, Disease-presents-Symptom, Disease-resembles-Disease, Disease-upregulates-Gene, Gene-covaries-Gene, Gene-interacts-Gene, Gene-participates-Biological Process, Gene-participates-Cellular Component, Gene-participates-Molecular Function, Gene-participates-Pathway, Gene-regulates-Gene, Pharmacologic Class-includes-Compound.
        Question: How many side effects does Caffeine have?
        Thought 1: The question is related to a compound node (Caffeine). We need to find the node in the graph and calculate the number of the node's 'Compound-causes-Side Effect' neighbors from the graph.
        Action 1: Degree[Retrieve[Caffeine], 'Compound-causes-Side Effect']
        Observation 1: The ID of this node is DB00201. 58
        Thought 2: TThe number of 'Compound-causes-Side Effect' neighbors are 58.
        Action 2: Finish[58]
            """,
    "biomedical_plan": """Definition of the graph: There are eleven types of nodes in the graph: Anatomy, Biological Process, Cellular Component, Compound, Disease, Gene, Molecular Function, Pathway, Pharmacologic Class, Side Effect, Symptom.
        Each node has name feature.
        There are these types of edges: Anatomy-downregulates-Gene, Anatomy-expresses-Gene, Anatomy-upregulates-Gene, Compound-binds-Gene, Compound-causes-Side Effect, Compound-downregulates-Gene, Compound-palliates-Disease, Compound-resembles-Compound, Compound-treats-Disease, Compound-upregulates-Gene, Disease-associates-Gene, Disease-downregulates-Gene, Disease-localizes-Anatomy, Disease-presents-Symptom, Disease-resembles-Disease, Disease-upregulates-Gene, Gene-covaries-Gene, Gene-interacts-Gene, Gene-participates-Biological Process, Gene-participates-Cellular Component, Gene-participates-Molecular Function, Gene-participates-Pathway, Gene-regulates-Gene, Pharmacologic Class-includes-Compound.
        Question: What compounds can be used to treat Crohn's disease? Please answer the compound names rather than IDs.
        Plan 1: The question is asking the compounds which can be used to treat a disease (Crohn's disease). We need to find the node in the graph and then find the compounds which can be used to treat this disease.
        Thought 1: We need to find the node in the graph.
        Action 1: Retrieve[Crohn's disease]
        Observation 1: The ID of this node is DOID:8778.
        Plan 2: The question is asking the compounds which can be used to treat a disease (Crohn's disease). We have the ID of this node. Thus, we need to find the compounds which can be used to treat this disease.
        Thought 2: We need to check the node's 'Compound-treats-Disease' neighbor from the graph and get their names.
        Action 2: Feature[Neighbor[DOID:8778, Compound-treats-Disease], name]
        Observation 2: ['DB01014', 'DB00244', 'DB00795', 'DB00993', 'DB00635', 'DB01033'] Balsalazide, Balsalazide, Mesalazine, Sulfasalazine, Azathioprine, Prednisone, Mercaptopurine
        Plan 3: The question is asking the compounds which can be used to treat a disease (Crohn's disease). We get the answer: Balsalazide, Mesalazine, Sulfasalazine, Azathioprine, Prednisone, Mercaptopurine.
        Thought 3: The name of compounds are Balsalazide, Mesalazine, Sulfasalazine, Azathioprine, Prednisone, Mercaptopurine.
        Action 3: Finish[Balsalazide, Mesalazine, Sulfasalazine, Azathioprine, Prednisone, Mercaptopurine]
        
        Definition of the graph: There are eleven types of nodes in the graph: Anatomy, Biological Process, Cellular Component, Compound, Disease, Gene, Molecular Function, Pathway, Pharmacologic Class, Side Effect, Symptom.
        Each node has name feature.
        There are these types of edges: Anatomy-downregulates-Gene, Anatomy-expresses-Gene, Anatomy-upregulates-Gene, Compound-binds-Gene, Compound-causes-Side Effect, Compound-downregulates-Gene, Compound-palliates-Disease, Compound-resembles-Compound, Compound-treats-Disease, Compound-upregulates-Gene, Disease-associates-Gene, Disease-downregulates-Gene, Disease-localizes-Anatomy, Disease-presents-Symptom, Disease-resembles-Disease, Disease-upregulates-Gene, Gene-covaries-Gene, Gene-interacts-Gene, Gene-participates-Biological Process, Gene-participates-Cellular Component, Gene-participates-Molecular Function, Gene-participates-Pathway, Gene-regulates-Gene, Pharmacologic Class-includes-Compound.
        Question: What is the inchikey of Caffeine?
        Plan 1: The question is asking the inchikey of a compound node (Caffeine). We need to find the node in the graph and then get its inchikey.
        Thought 1:We need to find the node in the graph and check the node's 'inchikey' feature from the graph.
        Action 1: Feature[Retrieve[Caffeine], inchikey]
        Observation 1: The ID of this node is DB00201. InChIKey=RYYVLZVUVIJVGH-UHFFFAOYSA-N
        Plan 2: The question is asking the inchikey of a compound node (Caffeine). We get the answer: InChIKey=RYYVLZVUVIJVGH-UHFFFAOYSA-N
        Thought 2:  The inchikey of the node is InChIKey=RYYVLZVUVIJVGH-UHFFFAOYSA-N.
        Action 2: Finish[InChIKey=RYYVLZVUVIJVGH-UHFFFAOYSA-N]
        
        Definition of the graph: There are eleven types of nodes in the graph: Anatomy, Biological Process, Cellular Component, Compound, Disease, Gene, Molecular Function, Pathway, Pharmacologic Class, Side Effect, Symptom.
        Each node has name feature.
        There are these types of edges: Anatomy-downregulates-Gene, Anatomy-expresses-Gene, Anatomy-upregulates-Gene, Compound-binds-Gene, Compound-causes-Side Effect, Compound-downregulates-Gene, Compound-palliates-Disease, Compound-resembles-Compound, Compound-treats-Disease, Compound-upregulates-Gene, Disease-associates-Gene, Disease-downregulates-Gene, Disease-localizes-Anatomy, Disease-presents-Symptom, Disease-resembles-Disease, Disease-upregulates-Gene, Gene-covaries-Gene, Gene-interacts-Gene, Gene-participates-Biological Process, Gene-participates-Cellular Component, Gene-participates-Molecular Function, Gene-participates-Pathway, Gene-regulates-Gene, Pharmacologic Class-includes-Compound.
        Question: How many side effects does Caffeine have?
        Plan 1: The question is asking the number of side effects of a compound node (Caffeine). We need to find the node in the graph and then get the number of the node's Compound-causes-Side Effect.
        Thought 1: The question is related to a compound node (Caffeine). We need to find the node in the graph and calculate the number of the node's 'Compound-causes-Side Effect' neighbors from the graph.
        Action 1: Degree[Retrieve[Caffeine], 'Compound-causes-Side Effect']
        Observation 1: The ID of this node is DB00201. 58
        Plan 2: The question is asking the number of side effects of a compound node (Caffeine). We get the answer: 58.
        Thought 2: TThe number of 'Compound-causes-Side Effect' neighbors are 58.
        Action 2: Finish[58]
            """,
}