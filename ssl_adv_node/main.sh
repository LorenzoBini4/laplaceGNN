# #### JOB ARRAY ###
python -u run_adv_node.py --flagfile=config/coauthor-cs.cfg > laplaceGNN.coauthor-cs.out 2> laplaceGNN.coauthor-cs.err
# python -u run_adv_node.py --flagfile=config/amazon-computers.cfg > laplaceGNN.amazon-computers.out 2> laplaceGNN.amazon-computers.err
# python -u run_adv_node.py --flagfile=config/amazon-photos.cfg > laplaceGNN.amazon-photos.out 2> laplaceGNN.amazon-photos.err
# python -u run_adv_node.py --flagfile=config/coauthor-physics.cfg > laplaceGNN.coauthor-physics.out 2> laplaceGNN.coauthor-physics.err
# python -u run_adv_node.py --flagfile=config/wiki-cs.cfg > laplaceGNN.wiki-cs.out 2> laplaceGNN.wiki-cs.err
# python -u run_adv_node.py --flagfile=config/ogbn-arxiv.cfg > laplaceGNN.ogbn-arxiv.out 2> laplaceGNN.ogbn-arxiv.err
# python -u run_adv_node.py --flagfile=config/pubmed.cfg > laplaceGNN.pubmed.out 2> laplaceGNN.pubmed.err
# python -u run_adv_node.py --flagfile=config/ogbn-papers.cfg > laplaceGNN.ogbn-papers.out 2> laplaceGNN.ogbn-papers.err

################## TRANSFER LEARNING ##################
# python -u run_adv_node_ppi.py --flagfile=config/ppi.cfg > laplaceGNN-ppi.o 2> laplaceGNN-ppi.e
# python -u run_adv_node_ppi.py --flagfile=config/zinc.cfg > laplaceGNN-zinc.o 2> laplaceGNN-zinc.e
