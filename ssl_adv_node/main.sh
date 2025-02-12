# #### JOB ARRAY ###
python -u run_adv_node.py --flagfile=config/coauthor-cs.cfg > laplacian.coauthor-cs.out 2> laplacian.coauthor-cs.err
# python -u run_adv_node.py --flagfile=config/amazon-computers.cfg > laplacian.amazon-computers.out 2> laplacian.amazon-computers.err
# python -u run_adv_node.py --flagfile=config/amazon-photos.cfg > laplacian.amazon-photos.out 2> laplacian.amazon-photos.err
# python -u run_adv_node.py --flagfile=config/coauthor-physics.cfg > laplacian.coauthor-physics.out 2> laplacian.coauthor-physics.err
# python -u run_adv_node.py --flagfile=config/wiki-cs.cfg > laplacian.wiki-cs.out 2> laplacian.wiki-cs.err
# python -u run_adv_node.py --flagfile=config/ogbn-arxiv.cfg > laplacian.ogbn-arxiv.out 2> laplacian.ogbn-arxiv.err
# python -u run_adv_node.py --flagfile=config/pubmed.cfg > laplacian.pubmed.out 2> laplacian.pubmed.err
# python -u run_adv_node.py --flagfile=config/ogbn-papers.cfg > laplacian.ogbn-papers.out 2> laplacian.ogbn-papers.err

################## TRANSFER LEARNING ##################
# python -u run_adv_node_ppi.py --flagfile=config/ppi.cfg > laplacian-ppi.o 2> laplacian-ppi.e
# python -u run_adv_node_ppi.py --flagfile=config/zinc.cfg > laplacian-zinc.o 2> laplacian-zinc.e
