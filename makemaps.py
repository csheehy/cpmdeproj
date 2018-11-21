import map

tt =   ['deriv', 'TR2.0', 'TR2.0', 'TR2.0+pol', 'TR2.0+pol', 'TR6.0', 'TR6.0+pol', 'TR10.0', 'TR10.0+pol', 'TR20.0', 'TR20.0+pol', 'TR20.0+pol']
dpdk = ['perdk', 'perdk', 'alldk', 'perdk',     'perdk',     'perdk', 'perdk',     'perdk',  'perdk',      'perdk',  'perdk',       'alldk']
cpm  = ['lr',    'lr',    'lr',    'lr',        'perpix',    'lr',    'lr',        'lr',     'lr',         'lr',     'lr',         'lr']


alpha = [1,0]


for a in alpha:

    for st in ['sig','noi','TnoP','signoi']:    

        for t,d,c in zip(tt,dpdk,cpm):

            dir = '004_'+t+'_alpha'+np.str(a)+'_cpm'+c+'_'+d+'/'

            m = map.map(dir+st+'_*.npz')
            m.save()

