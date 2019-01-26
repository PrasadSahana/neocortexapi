﻿////////////////////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2015 Frankfurt University of Applied Sciences / daenet GmbH
//
// Permission is hereby granted, free of charge, to any person
// obtaining a copy of this software and associated documentation
// files (the "Software"), to deal in the Software without
// restriction, including without limitation the rights to use,
// copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following
// conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.
////////////////////////////////////////////////////////////////////////////////////////////////

using System;
using System.Collections.Generic;
using System.Text;

namespace NeoCortexApi.Entities
{

    /// <summary>
    /// Helper entity for column bursting.
    /// </summary>
    public class BurstingTupple
    {
        
        public BurstingTupple(IList<Cell> cells, Cell bestCell)
        {
            this.Cells = cells;
            this.BestCell = bestCell;
        }

        /// <summary>
        ///  List of the processed column's cells.
        /// </summary>
        public IList<Cell> Cells { get; set; }


        /// <summary>
        /// Best rated cell.
        /// </summary>
        public Cell BestCell { get; set; }
    }
}
