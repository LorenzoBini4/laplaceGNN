# #### JOB ARRAY ###
python -u -m ssl_adv_node.run_adv_node --flagfile=config_node/coauthor-cs.cfg > run-coauthor-cs.out 2> run-coauthor-cs.err
# python -u -m ssl_adv_node.run_adv_node --flagfile=config_node/amazon-computers.cfg > run-amazon-computers.out 2> run-amazon-computers.err
# python -u -m ssl_adv_node.run_adv_node --flagfile=config_node/amazon-photos.cfg > run-amazon-photos.out 2> run-amazon-photos.err
# python -u -m ssl_adv_node.run_adv_node --flagfile=config_node/coauthor-physics.cfg > run-coauthor-physics.out 2> run-coauthor-physics.err
# python -u -m ssl_adv_node.run_adv_node --flagfile=config_node/wiki-cs.cfg > run-wiki-cs.out 2> run-wiki-cs.err
# python -u -m ssl_adv_node.run_adv_node --flagfile=config_node/ogbn-arxiv.cfg > run-ogbn-arxiv.out 2> run-ogbn-arxiv.err
# python -u -m ssl_adv_node.run_adv_node --flagfile=config_node/ogbn-papers.cfg > run-ogbn-papers.out 2> run-ogbn-papers.err

################## TRANSFER LEARNING ##################
# python -u -m ssl_adv_node.run_adv_node_ppi --flagfile=config_transfer/ppi.cfg > run-ppi.o 2> run-ppi.e
# python -u -m ssl_adv_node.run_adv_node_ppi --flagfile=config_transfer/zinc.cfg > run-zinc.o 2> run-zinc.e
